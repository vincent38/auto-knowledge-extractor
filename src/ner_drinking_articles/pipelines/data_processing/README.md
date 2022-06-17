# Pipeline data_processing

> *Note:* This pipeline should only be run through the Streamlit app. Start it using the command 'streamlit run main.py' on root folder.

## Overview

This pipeline handles the different tasks for the knowledge database build.

It is designed to be started from a streamlit application only, as it exchanges data on a real-time basis for user inputting and visualizations.

## Pipeline inputs

No input, the pipeline gathers data from the predefined set of URLs passed by streamlit.

## Pipeline outputs

The pipeline transfers several sets of nodes and edges, as well as a dictionnary of entities found during the NER task and sentiments processed during the SA task.
These are sent to the main streamlit application via user session variables, and only meant for visualization purposes.