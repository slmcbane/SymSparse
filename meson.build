project('SymSparse', ['cpp'], version: '1.0.1')

smallvector = subproject('SmallVector').get_variable('smallvector_dep')

symsparse_dep = declare_dependency(include_directories : include_directories('.'),
  dependencies : smallvector)

