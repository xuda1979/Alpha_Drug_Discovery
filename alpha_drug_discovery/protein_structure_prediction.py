import subprocess
import os

def predict_protein_structure(sequence, output_dir):
    """
    Run AlphaFold to predict the protein structure based on the sequence.

    Parameters:
    sequence (str): Protein sequence.
    output_dir (str): Directory to save the predicted structure.

    Returns:
    None
    """
    try:
        # Validate and create the FASTA file for the input sequence
        if not sequence.isalpha() or not sequence.isupper():
            raise ValueError("Invalid protein sequence. Must contain only uppercase letters.")
        
        os.makedirs(output_dir, exist_ok=True)
        fasta_file = os.path.join(output_dir, f"{sequence}.fasta")
        with open(fasta_file, 'w') as f:
            f.write(f">sequence\n{sequence}\n")

        # Run AlphaFold with the created FASTA file
        subprocess.run([
            "alphafold",
            "--fasta_paths", fasta_file,
            "--output_dir", output_dir
        ], check=True)
        
        print(f"Protein structure predicted and saved in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during AlphaFold execution: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
