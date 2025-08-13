from pathlib import Path

input_files_folder = Path(
    r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\data3')  # make the path a env variable


# create a excel data with 3 columns: filename, project, dijk
# fill the columnns with the files in the input_files_folder

def generate_template_master_table(input_folder: Path):
    data = []
    for file in input_folder.glob('*.xlsm'):
        # Extract project and dijk from the filename
        parts = file.stem.split('_')
        if len(parts) >= 2:
            project = file.stem.rsplit('_', 1)[-1]
            data.append({
                'filename': file.name,
                'project': project,
                'dijk': ''
            })

    # Create a DataFrame
    import pandas as pd
    df = pd.DataFrame(data)

    df.sort_values(by=['project'], inplace=True)

    # Save to Excel
    output_file = input_folder / 'master_table.xlsx'
    df.to_excel(output_file, index=False)
    print(f'Master table template saved to {output_file}')

if __name__ == "__main__":
    generate_template_master_table(input_files_folder)
