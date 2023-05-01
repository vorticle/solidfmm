AC_INIT([solidfmm],[1.5])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror nostdinc dist-xz subdir-objects])
AM_SILENT_RULES([yes])
AM_PROG_AR

AC_LANG([C++])
AC_PROG_CC
AC_PROG_CXX

AX_CXX_COMPILE_STDCXX(20,[noext],[mandatory])
AX_PTHREAD
PKG_CHECK_MODULES([HWLOC],[hwloc],[],AC_MSG_FAILURE([[

Starting with solidfmm-1.5, solidfmm depends on hwloc.
You can download hwloc from here: https://www.open-mpi.org/projects/hwloc/

]]))

LIBS="$PTHREAD_LIBS $HWLOC_LIBS $LIBS"
AM_CPPFLAGS="-I$srcdir $PTHREAD_CFLAGS $HWLOC_CFLAGS $AM_CPPFLAGS"
AC_SUBST([AM_CPPFLAGS])
AC_SUBST([LIBS])
PKG_INSTALLDIR

AC_CONFIG_HEADERS([solidfmm/config.h])
AC_CONFIG_FILES([Makefile solidfmm.pc])

LT_INIT
AC_OUTPUT

