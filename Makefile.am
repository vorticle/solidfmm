ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = doc/README doc/solidfmm.tex doc/literature.bib doc/solidfmm.pdf   \
             m4/ax_cxx_compile_stdcxx.m4 m4/ax_pthread.m4 m4/pkg.m4            \
             kirchhart_pubkey.gpg solidfmm.pc.in

pkgconfig_DATA = solidfmm.pc

lib_LTLIBRARIES = solidfmm/libsolidfmm.la
pkginclude_HEADERS = solidfmm/solid.hpp solidfmm/harmonics.hpp                 \
    solidfmm/translations.hpp solidfmm/handles.hpp solidfmm/sphere.hpp         

noinst_HEADERS = solidfmm/random.hpp solidfmm/stopwatch.hpp                    \
    solidfmm/swap_matrix.hpp solidfmm/microkernel.hpp                          \
    solidfmm/microkernel_avx.hpp solidfmm/microkernel_avx512f.hpp              \
    solidfmm/microkernel_generic.hpp solidfmm/microkernel_test.hpp             \
    solidfmm/microkernel_armv8a.hpp solidfmm/operator_data.hpp                 \
    solidfmm/threadlocal_buffer.hpp

solidfmm_libsolidfmm_la_SOURCES = solidfmm/solid.cpp solidfmm/harmonics.cpp    \
    solidfmm/handles.cpp solidfmm/sphere.cpp  solidfmm/swap_matrix.cpp         \
    solidfmm/operator_data.cpp solidfmm/threadlocal_buffer.cpp                 \
    solidfmm/microkernel.cpp solidfmm/microkernel_avx.cpp                      \
    solidfmm/microkernel_avx512f.cpp solidfmm/microkernel_generic.cpp          \
    solidfmm/microkernel_armv8a.cpp solidfmm/microkernel_test.cpp              \
    solidfmm/translations.cpp solidfmm/multithreaded_translator.cpp

solidfmm_libsolidfmm_la_LDFLAGS = -version-info 4:0:4



# Test Programs.
noinst_PROGRAMS = tests/test_swap_matrix tests/test_rotation tests/test_swap   \
    tests/test_m2l tests/test_m2m tests/test_l2l tests/test_microkernel        \
    tests/benchmark_m2l tests/benchmark_m2l_naive tests/benchmark_m2m          \
    tests/benchmark_l2l tests/benchmark_microkernel                            \
    tests/benchmark_multithreaded_translator

tests_test_swap_matrix_SOURCES  = tests/test_swap_matrix.cpp
tests_test_swap_matrix_LDADD    = $(top_builddir)/solidfmm/libsolidfmm.la 

tests_test_rotation_SOURCES = tests/test_rotation.cpp
tests_test_rotation_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la 

tests_test_swap_SOURCES = tests/test_swap.cpp
tests_test_swap_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_test_m2l_SOURCES = tests/test_m2l.cpp
tests_test_m2l_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_test_m2m_SOURCES = tests/test_m2m.cpp
tests_test_m2m_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_test_l2l_SOURCES = tests/test_l2l.cpp
tests_test_l2l_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_test_microkernel_SOURCES = tests/test_microkernel.cpp
tests_test_microkernel_LDADD = $(top_builddir)/solidfmm/libsolidfmm.la

tests_benchmark_microkernel_SOURCES = tests/benchmark_microkernel.cpp
tests_benchmark_microkernel_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_benchmark_m2l_SOURCES = tests/benchmark_m2l.cpp
tests_benchmark_m2l_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_benchmark_m2l_naive_SOURCES = tests/benchmark_m2l_naive.cpp
tests_benchmark_m2l_naive_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la 

tests_benchmark_m2m_SOURCES = tests/benchmark_m2m.cpp
tests_benchmark_m2m_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_benchmark_l2l_SOURCES = tests/benchmark_l2l.cpp
tests_benchmark_l2l_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

tests_benchmark_multithreaded_translator_SOURCES = tests/benchmark_multithreaded_translator.cpp
tests_benchmark_multithreaded_translator_LDADD   = $(top_builddir)/solidfmm/libsolidfmm.la

