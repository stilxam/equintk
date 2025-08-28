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
        # Read the hostname and remove the trailing newline.
        # This is an impure operation but common for personal configurations.
        # It allows the flake to adapt to the machine it's running on.

        # Set a boolean to true only if the hostname matches your desktop.

        # Configure nixpkgs, enabling CUDA support only when cudaEnabled is true.
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

        # Conditionally define the cudatoolkit package.
        # It will be 'null' on systems without CUDA.
        cudatoolkit = pkgs.cudatoolkit;

        # Conditionally create the environment hook for CUDA.
        # It will be an empty string if CUDA is not enabled.
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
        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          # Conditionally add CUDA toolkit to the shell's build inputs.
          # pkgs.lib.optionals adds the list of packages only if cudaEnabled is true. [2]
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
