AC_INIT([solidfmm],[1.0])
AC_CANONICAL_SYSTEM
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror nostdinc dist-xz])
AM_SILENT_RULES([yes])
AM_PROG_AR

AC_LANG([C++])
AC_PROG_CC
AC_PROG_CXX

AX_CXX_COMPILE_STDCXX_17([noext],[mandatory])

AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([solidfmm/config.h])
AC_CONFIG_FILES([Makefile
                 solidfmm/makefile
                 tests/makefile])

LT_INIT
AC_OUTPUT

