/*
 * Copyright (C) 2023 Matthias Kirchhart
 *
 * This file is part of solidfmm, a C++ library of operations on the solid
 * harmonics for use in fast multipole methods.
 *
 * solidfmm is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * solidfmm is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * solidfmm; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#include <solidfmm/translations.hpp>

#include <solidfmm/solid.hpp>
#include <solidfmm/stopwatch.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

#include <bit>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <limits>
#include <cstdint>
#include <barrier>
#include <exception>

#include <hwloc.h>

using namespace std;

namespace solidfmm
{

namespace // Anonymous.
{

// An implementation of NASAM, a mixing function by Pelle Evensen.
// https://mostlymangling.blogspot.com/2020/01/nasam-not-another-strange-acronym-mixer.html
//
// This is a pseudorandom, bijective permutation of the uint64_t numbers. 
// We use this function to ensure that translations are evenly distributed
// among worker threads.
//
// In personal communication, Pelle kindly told me to consider this mixing
// function as public domain.
inline uint64_t nasam( uint64_t x ) noexcept
{
    x ^= rotr(x,25) ^ rotr(x,47);
    x *= 0x9E6C63D0676A9A99UL;
    x ^= x >> 23 ^ x >> 51;
    x *= 0x9E6D62D06F6A9A9BUL;
    x ^= x >> 23 ^ x >> 51;

    return x;
}

inline uint64_t nasam( const void *const x ) noexcept
{
    return nasam(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x)));
}

// RAII Type to join threads on destruction.
struct guarded_thread: thread
{
    using thread::thread;

    guarded_thread( const guarded_thread  &rhs ) = delete;
    guarded_thread(       guarded_thread &&rhs ) noexcept
    {
        thread::operator=(std::move(rhs));
    }

    guarded_thread& operator=( const guarded_thread  &rhs ) = delete;
    guarded_thread& operator=(       guarded_thread &&rhs ) noexcept
    {
        thread::operator=(std::move(rhs));
        return *this;
    }

    ~guarded_thread()
    {
        if (joinable()) join(); 
    }
};

// RAII Type to manage hwloc_topology_t
// Either creates a new one and frees it on destruction,
// or it simply stores a given pointer which is *not* freed on destruction.
class topology_handle
{
private:
    bool owner;
    hwloc_topology_t t;

public:
    topology_handle()
    {
        int err;
        err = hwloc_topology_init(&t);
        if ( err )
            throw runtime_error { "solidfmm::multithreaded_translator: "
                                  "Could not initialise hwloc topology." };

        err = hwloc_topology_load(t);
        if ( err )
        {
            hwloc_topology_destroy(t);
            throw runtime_error { "solidfmm::multithreaded_translator: "
                                  "Could not detect hwloc topology." };
        }

        owner = true;
    }

    topology_handle( hwloc_topology_t p )
    {
        if ( p == nullptr )
            throw runtime_error { "solidfmm::multithreaded_translator: "
                                  "NULL hwloc_topology_t passed to constructor." };

        if ( hwloc_topology_abi_check(p) )
            throw runtime_error { "solidfmm::multithreaded_translator: "
                                  "hwloc_topology_t binary incompatible." };

        owner = false;
        t = p;
    }

    ~topology_handle()
    {
        if ( owner )
        {
            hwloc_topology_destroy(t);
        }
    }


    topology_handle( const topology_handle  &rhs ) = delete;
    topology_handle(       topology_handle &&rhs ) = delete;
    topology_handle& operator=( const topology_handle  &rhs ) = delete;
    topology_handle& operator=(       topology_handle &&rhs ) = delete;


    operator hwloc_topology_t() { return t; };
    hwloc_topology_t      get() { return t; };
};

// Needed for barrier for thread synchronisation
struct completion_function
{
    atomic_flag *flag;
    void operator()() noexcept { flag->clear( memory_order_release ); }
};

using barrier_t = std::barrier<completion_function>;



void set_thread_affinity( hwloc_topology_t topology,
                          hwloc_cpuset_t   cpuset )
{
    int err;
    err = hwloc_set_cpubind( topology, cpuset, HWLOC_CPUBIND_THREAD | 
                                               HWLOC_CPUBIND_STRICT );
    if ( err )
    {
        // Try without strict.
        err = hwloc_set_cpubind( topology, cpuset, HWLOC_CPUBIND_THREAD );
        if ( err )
            throw runtime_error { "solidfmm::multithreaded_translator:constructor "
                                  "Could not bind thread given cpuset." };
    }

    // Try to set first-touch policy. Failure is tolerable, so we do not
    // check the error-codes.
    // TODO: Have threadlocal_buffer use hwloc allocations instead.
    hwloc_set_membind( topology, cpuset, HWLOC_MEMBIND_NEXTTOUCH,
                                         HWLOC_MEMBIND_THREAD ); 
}



enum class job_type { idle, die, benchmark, m2m, m2l, l2l };

template <typename real>
struct worker_data
{
    using item = translation_info<real>;
    const atomic_flag    *signal;
          barrier_t      *sync;
          atomic_flag    *errflag;
          exception_ptr  *ex;

    const job_type       *job;
    const item           *const *begin, *const *end;

    const operator_data<real> *op;

    uint64_t min_hash, max_hash;
    real     throughput;  
};

template <typename real>
void worker_main( worker_data<real> *data,
                  hwloc_topology_t topology, hwloc_cpuset_t cpuset,
                  atomic_flag *init )
{
    using item = translation_info<real>;
    threadlocal_buffer<real>      buf; // Noexcept.
    vector<item>             item_buf; // Noexcept.
    try
    {
        set_thread_affinity(topology,cpuset);
        item_buf.resize(1337);
        buf = threadlocal_buffer<real> { *(data->op) };
    }
    catch ( ... )
    {
        bool tmp = data->errflag->test_and_set();
        if ( tmp == false )
        {
            *(data->ex) = current_exception();
        }
        init->test_and_set();
        init->notify_all();
        return;
    }

    init->test_and_set();
    init->notify_all();

    while ( true )
    {
        data->signal->wait(false,memory_order_acquire);
        if ( *(data->job) == job_type::die ) return;

        try
        { 
            work(data,buf,item_buf); 
        }
        catch ( ... )
        {
            if ( data->errflag->test_and_set() )
                *(data->ex) = current_exception();
        }
        data->sync->arrive_and_wait();
    }
}

template <typename real>
void work( worker_data<real> *data, threadlocal_buffer<real> &buf,
           vector<translation_info<real>> &item_buf )
{
    switch ( *(data->job) )
    {
    case job_type::die:  [[fallthrough]];
    case job_type::idle: return; break;
    case job_type::m2m:  work_m2m(data,buf,item_buf); break;
    case job_type::m2l:  work_m2l(data,buf,item_buf); break;
    case job_type::l2l:  work_l2l(data,buf,item_buf); break;
    case job_type::benchmark: measure_throughput(data,buf,item_buf); break;
    }
}

template <typename real>
void work_m2m( worker_data<real> *data, threadlocal_buffer<real> &buf,
               vector<translation_info<real>> &item_buf ) noexcept
{
    using item = translation_info<real>;
    size_t Nbuf = 0;
    for ( const item *i = *(data->begin); i != *(data->end); ++i )
    {
        uint64_t val = nasam(i->target);
        if ( data->min_hash <= val && val <= data->max_hash )
        {
            item_buf[Nbuf++] = *i;
            if ( Nbuf == item_buf.size() )
            {
                m2m_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
                Nbuf = 0;
            }
        }
    }

    if ( Nbuf ) m2m_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
}

template <typename real>
void work_m2l( worker_data<real> *data, threadlocal_buffer<real> &buf,
               vector<translation_info<real>> &item_buf ) noexcept
{
    using item = translation_info<real>;
    size_t Nbuf = 0;
    for ( const item *i = *(data->begin); i != *(data->end); ++i )
    {
        uint64_t val = nasam(i->target);
        if ( data->min_hash <= val && val <= data->max_hash )
        {
            item_buf[Nbuf++] = *i;
            if ( Nbuf == item_buf.size() )
            {
                m2l_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
                Nbuf = 0;
            }
        }
    }

    if ( Nbuf ) m2l_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
}

template <typename real>
void work_l2l( worker_data<real> *data, threadlocal_buffer<real> &buf,
               vector<translation_info<real>> &item_buf ) noexcept
{
    using item = translation_info<real>;
    size_t Nbuf = 0;
    for ( const item *i = *(data->begin); i != *(data->end); ++i )
    {
        uint64_t val = nasam(i->target);
        if ( data->min_hash <= val && val <= data->max_hash )
        {
            item_buf[Nbuf++] = *i;
            if ( Nbuf == item_buf.size() )
            {
                l2l_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
                Nbuf = 0;
            }
        }
    }

    if ( Nbuf ) l2l_unchecked( *(data->op), buf, item_buf.data(), item_buf.data() + Nbuf );
}

template <typename real>
void measure_throughput( worker_data<real> *data, threadlocal_buffer<real> &buf,
                         vector<translation_info<real>> &item_buf )
{
    using item = translation_info<real>;
    solid<real> blank( data->op->order() );
    vector<solid<real>> M( item_buf.size(), blank );
    vector<solid<real>> L( item_buf.size(), blank );

    for ( size_t i = 0; i < item_buf.size(); ++i )
    {
        item_buf[i].source = &M[i];
        item_buf[i].target = &L[i];
        item_buf[i].x = item_buf[i].y = item_buf[i].z = 1;
    }

    item *begin = item_buf.data();
    item *end   = item_buf.data() + item_buf.size();
   
    // Warm-up 
    m2l_unchecked( *(data->op), buf, begin, end );

    stopwatch<real> clock; 
    for ( size_t i = 0; i < 10; ++i )
        m2l_unchecked( *(data->op), buf, begin, end );
    real elapsed = clock.elapsed();
    data->throughput = static_cast<real>( size_t(10) * item_buf.size() ) / elapsed;
}

}

template <typename real>
class multithreaded_translator<real>::impl
{
public:
    impl() = delete;
    impl( size_t P );
    impl( size_t P, hwloc_topology_t topo,
                    hwloc_cpuset_t *begin, hwloc_cpuset_t *end );
    impl( const impl  &rhs ) = delete;
    impl(       impl &&rhs ) = delete;
    ~impl();

    impl& operator=( const impl  &rhs ) = delete;
    impl& operator=(       impl &&rhs ) = delete;

    void m2m( const item *begin, const item *end );
    void m2l( const item *begin, const item *end );
    void l2l( const item *begin, const item *end );

    void m2m_unchecked( const item *begin, const item *end ) noexcept;
    void m2l_unchecked( const item *begin, const item *end ) noexcept;
    void l2l_unchecked( const item *begin, const item *end ) noexcept;

    void schedule_equally() noexcept;
    void schedule_by_measuring_throughput();

private:
    void check( const item *begin, const item *end );

private:
    mutable mutex m;
    const operator_data<real> op;

    using barrier_ptr = unique_ptr<barrier_t>;
   
    exception_ptr  ex;
    job_type       job;
    atomic_flag    signal;
    barrier_ptr    sync;
    atomic_flag    errflag;
    const item     *begin, *end;

    vector<worker_data<real>> wdata;
    vector<guarded_thread>    threads;
};

template <typename real>
multithreaded_translator<real>::impl::impl( size_t P ):
op { P }
{
    topology_handle topology;
    int corelevel   = hwloc_get_type_or_below_depth(topology,HWLOC_OBJ_CORE);
    size_t nthreads = hwloc_get_nbobjs_by_depth(topology,corelevel);

    if ( nthreads == 0 )
        throw runtime_error { "solidfmm::multithreaded_translator::constructor: "
                              "Cannot run on zero threads." };

    wdata.resize(nthreads);
    threads.resize(nthreads);
    sync = barrier_ptr { new barrier_t(1 + nthreads,completion_function { &signal }) };

    using cpuset_ptr = unique_ptr<hwloc_bitmap_s,decltype(&hwloc_bitmap_free)>;
    try
    {
        for ( size_t i = 0; i < nthreads; ++i )
        {
            hwloc_obj_t obj    { hwloc_get_obj_by_depth(topology,corelevel,i) };
            cpuset_ptr  cpuset { hwloc_bitmap_dup(obj->cpuset), &hwloc_bitmap_free };
            if ( cpuset == nullptr ) throw std::bad_alloc {};
            hwloc_bitmap_singlify(cpuset.get());

            wdata[i].op      = &op;
            wdata[i].ex      = &ex;
            wdata[i].begin   = &begin;
            wdata[i].end     = &end;
            wdata[i].job     = &job;
            wdata[i].signal  = &signal;
            wdata[i].sync    = sync.get();
            wdata[i].errflag = &errflag;
            atomic_flag init;
            threads[i] = guarded_thread { worker_main<real>, &(wdata[i]),
                                          topology.get(), cpuset.get(), &init };
            init.wait(false);
            if ( errflag.test() )
                rethrow_exception(ex);
        }
    }
    catch ( ... )
    {
        job = job_type::die;
        signal.test_and_set(memory_order_release);
        signal.notify_all();
        throw;
    }

    schedule_equally();
}


template <typename real>
multithreaded_translator<real>::impl::impl( size_t P, hwloc_topology_t topo,
                                            hwloc_cpuset_t *cpus_begin, hwloc_cpuset_t *cpus_end ):
op { P }
{
    topology_handle topology { topo };
    size_t nthreads = cpus_end-cpus_begin;

    if ( nthreads == 0 )
        throw runtime_error { "solidfmm::multithreaded_translator::constructor: "
                              "Cannot run on zero threads." };

    wdata.resize(nthreads);
    threads.resize(nthreads);
    sync = barrier_ptr { new barrier_t(1 + nthreads,completion_function { &signal }) };

    try
    {
        for ( size_t i = 0; i < nthreads; ++i )
        {
            wdata[i].op      = &op;
            wdata[i].ex      = &ex;
            wdata[i].begin   = &begin;
            wdata[i].end     = &end;
            wdata[i].job     = &job;
            wdata[i].signal  = &signal;
            wdata[i].sync    = sync.get();
            wdata[i].errflag = &errflag;
            atomic_flag init;
            threads[i] = guarded_thread { worker_main<real>, &(wdata[i]),
                                          topology.get(), cpus_begin[i], &init };
            init.wait(false);
            if ( errflag.test() )
                rethrow_exception(ex);
        }
    }
    catch ( ... )
    {
        job = job_type::die;
        signal.test_and_set(memory_order_release);
        signal.notify_all();
        throw;
    }

    schedule_equally();
}

template <typename real>
multithreaded_translator<real>::impl::~impl()
{
    job = job_type::die;
    signal.test_and_set(memory_order_release);
    signal.notify_all();
}

template <typename real>
void multithreaded_translator<real>::impl::m2m_unchecked( const item *p_begin, const item *p_end ) noexcept
{
    lock_guard<mutex> lock { m };
    begin = p_begin;
    end   = p_end;
    job   = job_type::m2m;

    signal.test_and_set(memory_order_release); signal.notify_all();
    sync->arrive_and_wait();
}

template <typename real>
void multithreaded_translator<real>::impl::m2l_unchecked( const item *p_begin, const item *p_end ) noexcept
{
    lock_guard<mutex> lock { m };
    begin = p_begin;
    end   = p_end;
    job   = job_type::m2l;

    signal.test_and_set(memory_order_release); signal.notify_all();
    sync->arrive_and_wait();
}

template <typename real>
void multithreaded_translator<real>::impl::l2l_unchecked( const item *p_begin, const item *p_end ) noexcept
{
    lock_guard<mutex> lock { m };
    begin = p_begin;
    end   = p_end;
    job   = job_type::l2l;

    signal.test_and_set(memory_order_release); signal.notify_all();
    sync->arrive_and_wait();
}

template <typename real>
void multithreaded_translator<real>::impl::m2m( const item *p_begin, const item *p_end ) 
{
    check(p_begin,p_end);
    m2m_unchecked(p_begin,p_end);
}

template <typename real>
void multithreaded_translator<real>::impl::m2l( const item *p_begin, const item *p_end ) 
{
    check(p_begin,p_end);
    m2l_unchecked(p_begin,p_end);
}

template <typename real>
void multithreaded_translator<real>::impl::l2l( const item *p_begin, const item *p_end ) 
{
    check(p_begin,p_end);
    l2l_unchecked(p_begin,p_end);
}

template <typename real>
void multithreaded_translator<real>::impl::check( const item *p_begin, const item *p_end )
{
    for ( const item *i = p_begin; i != p_end; ++i )
    {
        if ( i->source->dimension() != i->target->dimension() )
            throw std::logic_error { "solidfmm::l2l<double>(): dimension mismatch." };

        if ( i->source->order() > op.order() || i->target->order() > op.order() )
            throw std::out_of_range { "solidfmm::l2l<double>(): orders exceeding operator data." };
    }
}


template <typename real>
void multithreaded_translator<real>::impl::schedule_by_measuring_throughput()
{
    lock_guard<mutex> lock { m };
    errflag.clear();
    job = job_type::benchmark;
    signal.test_and_set(memory_order_release); signal.notify_all();
    sync->arrive_and_wait();

    if ( errflag.test() )
    {   
        errflag.clear();
        rethrow_exception(ex);
    }

    size_t nthreads = threads.size();
    real total_throughput = 0;
    for ( size_t i = 0; i < nthreads; ++i )
        total_throughput += wdata[i].throughput;

    uint64_t curr = 0;
    constexpr real rmax = static_cast<real>(numeric_limits<uint64_t>::max());
    for ( size_t i = 0; i < nthreads; ++i )
    {
        wdata[i].min_hash = curr;     
        curr += static_cast<uint64_t>
        ( 
            (wdata[i].throughput/total_throughput)*rmax
        );
        wdata[i].max_hash = curr++;
    }
    wdata.back().max_hash = numeric_limits<uint64_t>::max();
}

template <typename real>
void multithreaded_translator<real>::impl::schedule_equally() noexcept
{
    lock_guard<mutex> lock { m };
    uint64_t nthreads = threads.size();

    constexpr uint64_t max = numeric_limits<uint64_t>::max();

    //(max%nthreads)+1, because there are max+1 possible values of an uint64_t.
    uint64_t mod  = max % nthreads + 1; 
    uint64_t step = max / nthreads;
    uint64_t curr = 0;
    for ( uint64_t i = 0; i < nthreads; ++i )
    {
        wdata[i].min_hash = curr;
        if ( i < mod ) curr +=  step;
        else           curr += (step-1);
        wdata[i].max_hash = curr++;
    }
}

template <typename real>
multithreaded_translator<real>::multithreaded_translator( size_t P ):
p { new impl { P } }
{}

template <typename real>
multithreaded_translator<real>::multithreaded_translator( size_t P, hwloc_topology *topology,
                                                    cpuset_t *begin, cpuset_t *end ):
p { new impl { P, topology, begin, end } }
{}

template <typename real>
multithreaded_translator<real>::~multithreaded_translator()
{
    delete p;
}

template <typename real>
multithreaded_translator<real>::
multithreaded_translator( multithreaded_translator &&rhs ) noexcept:
p { rhs.p }
{
    rhs.p = nullptr;
}

template <typename real>
multithreaded_translator<real>& multithreaded_translator<real>::
operator=( multithreaded_translator &&rhs ) noexcept
{
    if ( this != &rhs )
    {
        delete p;
        p = rhs.p;
        rhs.p = nullptr;
    }
    return *this;
}

template <typename real>
void multithreaded_translator<real>::m2m( const item *begin, const item *end ) const
{
    p->m2m(begin,end);
}

template <typename real>
void multithreaded_translator<real>::m2l( const item *begin, const item *end ) const
{
    p->m2l(begin,end);
}

template <typename real>
void multithreaded_translator<real>::l2l( const item *begin, const item *end ) const
{
    p->l2l(begin,end);
}

template <typename real>
void multithreaded_translator<real>::m2m_unchecked( const item *begin, const item *end ) const noexcept
{
    p->m2m_unchecked(begin,end);
}

template <typename real>
void multithreaded_translator<real>::m2l_unchecked( const item *begin, const item *end ) const noexcept
{
    p->m2l_unchecked(begin,end);
}

template <typename real>
void multithreaded_translator<real>::l2l_unchecked( const item *begin, const item *end ) const noexcept
{
    p->l2l_unchecked(begin,end);
}

template <typename real>
void multithreaded_translator<real>::schedule_equally_among_threads() noexcept
{
    p->schedule_equally();
}

template <typename real>
void multithreaded_translator<real>::schedule_by_measuring_throughput() 
{
    p->schedule_by_measuring_throughput();
}

// Explicit instantiations.
template class multithreaded_translator<float>;
template class multithreaded_translator<double>;

}

