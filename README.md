# CODEX_LateralDiffusion_PythonGUI
README for 31P CODEX Decay Curve Fitting GUI 

Overview

    This application provides a graphical user interface (GUI) for fitting 31P CODEX (Centerband-Only Detection of Exchange) solid state NMR data. It simulates the signal decay due to lateral diffusion (Dlat​) of phospholipids in a liposome sample caused by the recoupling of chemical shift anisotropy (CSA) during the CODEX pulse sequence. The code allows users to extract Dlat based on the experimental inputs​, and calculate error bars for the fitted values.

File Structure

    python_gui/
    │
    ├── main.py             # Initializes the Tkinter root and handles application closure.
    ├── states.py           # Contains GUI setup functions and initializes variables stored in the state dictionary.
    ├── functions.py        # Contains functions that manipulate the state dictionary and perform simulations and calculations.
    ├── requirements.txt    # Lists the project dependencies.

Prerequisites

    Python 3.x: Ensure you have Python 3 installed on your system.
    pip: The Python package installer.

Installation Steps

    1. Clone or Download the Repository

        Navigate to the directory where you want to place the project and clone the repository or download and extract it.

    2. Navigate to the Project Directory

        Open a terminal and navigate to the python_gui directory:

        cd path/to/python_gui

    3. Create a Virtual Environment (Optional but Recommended)

        python -m venv venv

        Activate the virtual environment:

            On Windows:

                venv\Scripts\activate

            On macOS/Linux:

                source venv/bin/activate

    4. Install Dependencies

        Install the required packages using requirements.txt:

            pip install -r requirements.txt

        Note: Replace pip with pip3 if necessary.

Running the Application

    After installing the dependencies, you can run the application:

        python main.py

    Alternatively, you can use an IDE like Visual Studio Code and use the run/debug features.

Functionality

    main.py

        Purpose: Initializes the Tkinter root window, handles the application closure, and starts the main event loop.
        Key Actions:
            Creates the root Tkinter window.
            Initializes the state dictionary to hold application data.
            Calls init_vars and init_gui from states.py to set up variables and GUI elements.
            Handles graceful shutdown of the application, ensuring threads are properly terminated.

    states.py

        Purpose: Sets up GUI elements and initializes variables stored in the state dictionary.
        Key Components:
            init_vars(state): Initializes variables such as default values for inputs and stores them in the state dictionary.
            init_gui(root, state): Builds the GUI layout, including frames, labels, entries, buttons, and canvases. Widgets are stored in the state dictionary.
            Event Bindings: Widgets like buttons are linked to functions defined in functions.py using the partial function from functools.

    functions.py

        Purpose: Contains functions that perform the core functionalities, including data import, simulation, optimization, error calculation, plotting, and message handling.
        Key Functions:
            select_codex_file(state): Opens a file dialog to select the CODEX decay data file.
            import_decay_data(state): Imports and plots the CODEX decay data.
            select_lsd_file(state): Opens a file dialog to select the liposome size distribution file.
            plot_lsd_curve(state): Imports and plots the liposome size distribution data.
            extract_gui_inputs(state): Retrieves and validates user inputs from the GUI.
            run_simulation_in_thread(state): Starts the fitting simulation in a separate thread.
            simulation_task(state): Performs the fitting loop and updates the GUI upon completion.
            fitting_loop(state): Runs the optimization over the CSA range to fit the experimental data.
            calculate_phase_matrix_vectorized(state, ...): Computes the phase matrix for CSA recoupling effects.
            data_simulation(state, ...): Simulates the NMR signal decay based on the model.
            dlat_optimization(state, ...): Optimizes Dlat​ to fit the experimental data.
            calculate_dlat_error_bars_with_csa_range(state, ...): Calculates and propagates errors to obtain error bars for Dlat​.
            update_gui_with_results(state, results): Updates plots and GUI elements with the simulation results.
            update_sres_plot(state, ...) and update_dlat_plot(state, ...): Plot the sum of squared residuals and Dlat​ vs. δCSA, respectively.
            send_message_to_box(state, message): Displays messages in the message box within the GUI.
            process_message_queue(state): Manages the message queue for thread-safe GUI updates.

Using the Application

    1. Import Data:
        Click on "Select File for CODEX Decay Signal" to choose your CODEX decay data file.
        Click on "Import Decay Data" to load and visualize the data.
        Click on "Select File for Liposome Size Distribution" to choose your liposome size distribution file.
        Click on "Import Distribution Data" to load and visualize the distribution.

    2. Set Parameters:
        Enter the experimental parameters in the "Experimental Parameters" frame.
        Enter simulation parameters in the "Simulation Parameters" frame.

    3. Run Fitting Simulation:
        Click on "Fit CODEX Curve" to start the fitting process.
        Monitor progress and messages in the "Message Box".

    4. Review Results:
        The fitted CODEX decay curve will be displayed alongside the experimental data.
        Additional plots (sres vs. δCSA and Dlat​ vs. δCSA) will be shown in the "Other Plots" frame.
        Error bars for Dlat​ will be calculated and displayed in the message box.

    5. Export Figures:
        Use the "Export Figure" button to save the CODEX decay curve plot.

    6. Stop Simulation:
        Click on "Stop Simulation" to interrupt the fitting process if needed.

Dependencies

    All required packages are listed in requirements.txt. The main dependencies include:

        numpy: Numerical computations.
        pandas: Data manipulation.
        matplotlib: Plotting and visualization.
        scipy: Optimization and interpolation.
        numba: Just-in-time compilation for performance optimization.
        tkinter: GUI framework (usually included with Python).
        queue: Thread-safe message handling.

    Note: The tkinter module is included with standard Python installations. If you're using a Python distribution that doesn't include it, you may need to install it via your system's package manager (e.g., sudo apt-get install python3-tk on Ubuntu).

    Ensure that you have compatible versions of these packages. It's recommended to use the latest stable versions unless specific version constraints are necessary due to compatibility.
    Important Notes

        Data Format:
            CODEX Decay Data: Two-column format with mixing times and normalized intensities (intensities normalized with respect to the value at the shortest mixing time, which is set to a value of 1).

            Liposome Size Distribution: Two-column format with radii and normalized number probabilities, where the sum over P(r) = 1.

        Error Handling: The application provides messages in the message box for errors and important events. Ensure that inputs are valid and files are correctly formatted.

        Performance Considerations: The simulation can be computationally intensive, especially with fine angle increments and small δCSA step sizes. Adjust parameters accordingly to balance precision and computation time. Recommended to use a minimum angle increment of 4.

Troubleshooting

    Import Errors: If you encounter import errors, double-check that all dependencies are installed and that you're using the correct Python environment.

    Threading Issues: The fitting simulation runs in a separate thread. If the application becomes unresponsive, ensure that the threading and message queue are functioning correctly.
