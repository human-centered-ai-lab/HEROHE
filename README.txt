Diagnostic- and Research-Institute of Pathology, Medical University of Graz
Preprocessing with QuPath:
- Create new project in QuPath
- Import slides into Project
- Run groovy script "QuPath/ProjectDetection.groovy" for Project ("Automate" -> "Show Script Editor" -> "Run" -> "Run for Project")
    - The output will be in the Project Folder

Data Preparation with Python:
- You have to update the "python_script/dataPreperation.py"
   - Line 8: Path to QuPath Project folder with the extraced dada ("PROJECT_BASE_DIR/detections")
   - Line 9: Database connection string
- Run the dataPreperation.py file

Predict slide label:
- You have to update the "python_script/XXX.py"
   - Line X: Database connection string
- Run XXX.py