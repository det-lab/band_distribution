project: band_distribution
summary: A total phonon energy and total photon energy calculator for dark matter searches
src_dir: src
exclude_dir: doc
output_dir: doc/html
preprocess: true
macro: FORD
preprocessor: gfortran -E
display: public
         protected
         private
source: true
graph: true
md_extensions: markdown.extensions.toc
coloured_edges: true
sort: permission-alpha
extra_mods: iso_fortran_env:https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
            iso_c_binding:https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fC_005fBINDING.html#ISO_005fC_005fBINDING
project_github: https://github.com/det-lab/band_distribution
project_download: https://github.com/det-lab/band_distribution/releases
author: Amy Roberts
github: https://github.com/pibion
print_creation_date: true
creation_date: %Y-%m-%d %H:%M %z
predocmark_alt: >
predocmark: <
docmark_alt:
docmark: !

{!README.md!}
