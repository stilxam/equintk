{
  description = "JAX+EQX flake for NTK computation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    unstable-nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, unstable-nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        unstable-pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312;

        cudatoolkit = pkgs.cudatoolkit;

        cudaEnvHook = ''
          export CUDA_HOME=${cudatoolkit}
          export CUDA_ROOT=${cudatoolkit}
          export LD_LIBRARY_PATH="${cudatoolkit.lib}/lib:${cudatoolkit}/lib:$LD_LIBRARY_PATH"
          export PATH="${cudatoolkit}/bin:$PATH"
          export CMAKE_PREFIX_PATH="${cudatoolkit}:$CMAKE_PREFIX_PATH"
        '';


        mainPythonPackages = ps: with ps; [
            cython
            pytest
            pip
            jax
            jaxlib
            equinox
            jaxtyping
            matplotlib
            seaborn
            notebook
        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
	    cudatoolkit
	    unstable-pkgs.gemini-cli
          ] ;

          shellHook = cudaEnvHook + ''
            export PYTHONPATH=${toString ./.}/$PYTHONPATH
            echo "CUDA toolkit available at: $CUDA_HOME"
          '';
        };
      });
}
