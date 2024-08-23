import os
import subprocess

def run_autodock_ligand(ligand_file, protein_file, output_dir):
    """
    Run AutoDock for ligand-protein docking simulation.

    Parameters:
    ligand_file (str): Path to the ligand file.
    protein_file (str): Path to the protein file.
    output_dir (str): Directory to save the docking results.

    Returns:
    None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        docking_command = f"autodock4 -p {protein_file} -l {ligand_file} -o {output_dir}/docking_output.dlg"
        result = subprocess.run(docking_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Docking completed. Results saved in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during docking: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
