{
  description = "Reassess toolbox devshell";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 python312Packages.pip neovim ffmpeg ];
        buildInputs = with pkgs; [ python312 ];
        NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.zlib
          pkgs.libGL
          pkgs.glib
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:${pkgs.glibc}/lib
          export NIX_LD=$(cat ${pkgs.stdenv.cc}/nix-support/dynamic-linker)
          export XDG_CONFIG_HOME=$HOME/.config  # Adjust if your config is in a specific directory
          export NVIM_CONFIG_DIR=$XDG_CONFIG_HOME/nvim
        '';
      };
    };
}
