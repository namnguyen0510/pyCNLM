The 27 MB of bundled classical MaxSAT competition solvers (BandHS,
FourierSAT, NuWLS, SATLike3.0, SPB-MaxSAT) were stripped from this
package to keep the zip small.  They are NOT needed to run the SGAT
GNN itself (--sgat-mode sgat).  To use the bundled classical solvers,
re-download them from the original SGAT-MS repo:

    git clone https://github.com/sotam2369/SGAT-MS.git
    cp SGAT-MS/solvers/*.zip third_party/SGAT-MS/solvers/

Then unzip each into its own directory.
