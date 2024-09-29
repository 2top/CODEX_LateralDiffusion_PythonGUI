import os
import numpy as np
import pandas as pd
import threading 

from tkinter import filedialog
from numba import njit, prange
from scipy.interpolate import UnivariateSpline
from scipy.stats import t
from scipy.optimize import minimize

import queue

def process_message_queue(state):
    try:
        while True:
            message = state['message_queue'].get_nowait()
            send_message_to_box(state, message)
    except queue.Empty:
        pass
    # Schedule this function to be called again after a short delay
    after_id = state['root'].after(100, process_message_queue, state)
    state['process_message_queue_after_id'] = after_id



def extract_gui_inputs(state):
    inputs = {}
    try:
        # Extract numerical inputs and convert them to floats
        inputs['delta_csa'] = float(state['delta_csa_entry'].get())
        inputs['mas_rate'] = float(state['mas_rate_entry'].get())
        inputs['pulses_180'] = float(state['pulses_180_entry'].get())
        inputs['freq_31p'] = float(state['freq_31p_entry'].get())
        inputs['temp'] = float(state['temp_entry'].get())
        inputs['viscosity'] = float(state['viscosity_entry'].get())
        inputs['dlat_guess'] = float(state['dlat_guess_entry'].get())
        inputs['angle_inc'] = float(state['angle_inc_entry'].get())
        inputs['csa_range'] = float(state['csa_range_entry'].get())
        inputs['csa_step'] = float(state['csa_step_entry'].get())

        # Validate inputs
        if inputs['delta_csa'] <= 0:
            raise ValueError("δCSA must be positive.")
        if inputs['mas_rate'] <= 0:
            raise ValueError("MAS rate must be positive.")
        if inputs['pulses_180'] < 1:
            raise ValueError("Number of 180 pulses must be greater than or equal to 1.")
        if not inputs['pulses_180'].is_integer():
            raise ValueError("Number of 180 pulses must be an integer.")
        inputs['pulses_180'] = int(inputs['pulses_180'])
        if inputs['pulses_180'] % 2 == 0:
            raise ValueError("Number of 180 pulses must be odd.")
        if inputs['freq_31p'] <= 0:
            raise ValueError("31P Frequency must be positive.")
        if inputs['temp'] <= 0:
            raise ValueError("Temperature (in Kelvin) must be positive.")
        if inputs['viscosity'] <= 0:
            raise ValueError("Viscosity must be positive.")
        if inputs['dlat_guess'] <= 0:
            raise ValueError("Initial guess for Dlat must be positive.")
        if inputs['angle_inc'] <= 0:
            raise ValueError("Angle increment must be positive.")
        if inputs['csa_range'] <= 0:
            raise ValueError("Value for the δCSA range must be positive.")
        if inputs['csa_step'] <= 0:
            raise ValueError("Value for the δCSA step range must be positive.")

    except ValueError as e:
        # Put the message into the queue
        state['message_queue'].put(f"Input Error: {e}")
        return None

    return inputs


def run_simulation_in_thread(state):
    # Initialize the message queue if not already initialized
    if 'message_queue' not in state:
        state['message_queue'] = queue.Queue()
    # Disable the "Fit CODEX Curve" button
    state['fit_codex_curve_button']['state'] = 'disabled'
    # Enable the "Stop Simulation" button
    state['stop_simulation_button']['state'] = 'normal'

    # Initialize the stop event
    state['simulation_stop_event'] = threading.Event()
    # Start the simulation in a new daemon thread
    simulation_thread = threading.Thread(target=simulation_task, args=(state,), daemon=True)
    simulation_thread.start()
    # Store the thread reference
    state['simulation_thread'] = simulation_thread


def simulation_task(state):
    try:
        # Run the fitting loop (simulation)
        results = fitting_loop(state)
        if results is not None:
            # Schedule the GUI update in the main thread
            state['root'].after(0, update_gui_with_results, state, results)
    except Exception as e:
        # Handle exceptions
        state['message_queue'].put(f"Simulation Error: {e}")
    finally:
        # Clear the stop event
        state['simulation_stop_event'].clear()
        # Remove the thread reference
        state['simulation_thread'] = None
        # Re-enable the buttons in the GUI
        state['root'].after(0, update_button_states, state)



def update_button_states(state):
    # Re-enable the "Fit CODEX Curve" button
    state['fit_codex_curve_button']['state'] = 'normal'
    # Disable the "Stop Simulation" button
    state['stop_simulation_button']['state'] = 'disabled'


def stop_simulation(state):
    """
    Function to stop the running simulation.
    """
    # Check if a simulation is running
    if 'simulation_stop_event' in state and state['simulation_stop_event'] is not None:
        # Set the stop event
        state['simulation_stop_event'].set()
        send_message_to_box(state, 'Simulation stop requested.')
    else:
        send_message_to_box(state, 'No simulation is currently running.')


def simulation_complete(state):
    # Redraw the canvas
    state['codex_canvas'].draw()

    # Update other plots if necessary
    # update_sres_plot(state, state['csaList'], state['sresListForCSAPlot'])
    # update_dlat_plot(state, state['csaList'], state['DlatListForCSAPlot'])

    send_message_to_box(state, 'CODEX Curve Fitting Completed.')

def select_codex_file(state):
        state['codex_file'] = filedialog.askopenfile(
            parent=state['root'],
            title="Select the CODEX file",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("CSV files", "*.csv")]
            )
        if state['codex_file'] is not None:
            state['codex_file_label'].config(text="CODEX File: " + state['codex_file'].name.split("/")[-1])
        else:
            state['codex_file_label'].config(text="CODEX File: ")



def select_lsd_file(state):
    state['lsd_file'] = filedialog.askopenfile(
        parent=state['root'],
        title="Select the Liposome Size Distribution file",
        filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
    if state['lsd_file'] is not None:
        state['lsd_file_label'].config(text="Liposome Size Distribution File: " + state['lsd_file'].name.split("/")[-1])
    else:
        state['lsd_file_label'].config(text="Liposome Size Distribution File: ")



def import_decay_data(state):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    plt.rcParams['font.family'] = 'Arial'

    # read in file for CODEX decay signal
    file_name = state['codex_file'].name
    file_ext = os.path.splitext(file_name)[1]

    if file_ext == '.xlsx':
        state['decay_data'] = pd.read_excel(file_name, header=None)
    elif file_ext == '.csv':
        state['decay_data'] = pd.read_csv(file_name, header=None)
    elif file_ext == '.txt':
        state['decay_data'] = pd.read_csv(file_name, sep='\t', header=None)
    else:
        raise ValueError("File type not supported")

    x_vals = state['decay_data'].iloc[:, 0]
    y_vals = state['decay_data'].iloc[:, 1]
    fig, ax = plt.subplots(figsize=(4,3))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.15)

    ax.set_ylim(0, 1.05)
    ax.set_title('Codex Signal vs. Mixing Time')
    ax.set_xlabel('Mixing Time (s)')
    ax.set_ylabel('CODEX Signal')
    
    ax.scatter(x_vals, y_vals, color='black')

    # set_width, set_height = fig.get_size_inches() * fig.dpi * 0.8
    # print(fig.get_size_inches())

    state['codex_canvas'] = FigureCanvasTkAgg(fig, master=state['codex_placeholder_canvas'])
    # state['codex_canvas'].get_tk_widget().configure(width=set_width, height=set_height)
    state['codex_canvas'].get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    state['codex_canvas'].draw()

    state['codex_ax'] = ax
    
    #send message to box that data has been imported
    send_message_to_box(state, f"Data file {state['codex_file'].name.split('/')[-1]} Imported!")



def plot_lsd_curve(state):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    plt.rcParams['font.family'] = 'Arial'

    # read in file for Liposome Size Distribution
    file_name = state['lsd_file'].name
    file_ext = os.path.splitext(file_name)[1]

    if file_ext == '.xlsx':
        state['lsd_data'] = pd.read_excel(file_name, header=None)
    elif file_ext == '.csv':
        state['lsd_data'] = pd.read_csv(file_name, header=None)
    elif file_ext == '.txt':
        state['lsd_data'] = pd.read_csv(file_name, sep='\t', header=None)
    else:
        raise ValueError("File type not supported")
    
    x_vals = state['lsd_data'].iloc[:, 0]
    y_vals = state['lsd_data'].iloc[:, 1]
    
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    state['lsd_canvas'] = FigureCanvasTkAgg(fig, master=state['lsd_placeholder_canvas'])
    state['lsd_canvas'].get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    state['lsd_canvas'].draw()

    ax.plot(x_vals, y_vals, color='black')
    ax.set_title('Liposome Size Distribution')
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Number Probability')

    send_message_to_box(state, f"Data file {state['lsd_file'].name.split('/')[-1]} Imported!")


def export_codex_canvas(state):
    """
    Exports the codex_canvas as a figure in various formats.
    """
    # Check if the codex_ax exists
    if 'codex_ax' not in state or state['codex_ax'] is None:
        send_message_to_box(state, "No CODEX decay curve to export. Please run a simulation first.")
        return
    
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[
            ("PDF files", "*.pdf"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("All files", "*.*")
        ],
        title="Save CODEX Decay Curve"
    )

    if file_path:
        try:
            # Determine the format based on the file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext == '.pdf':
                file_format = 'pdf'
            elif ext in ['.png']:
                file_format = 'png'
            elif ext in ['.jpg', '.jpeg']:
                file_format = 'jpeg'
            else:
                # Default to PDF if unknown extension
                file_format = 'pdf'
                file_path += '.pdf'

            # Save the figure
            state['codex_ax'].figure.savefig(file_path, format=file_format)
            send_message_to_box(state, f"Figure saved successfully at {file_path}")
        except Exception as e:
            send_message_to_box(state, f"Error saving figure: {e}")
    else:
        send_message_to_box(state, "Save operation cancelled.")



# def clear_message_box(state):
#     state['message_box'].config(state="normal")
#     state['message_box'].delete('1.0', 'end')
#     state['message_box'].config(state="disabled")



def send_message_to_box(state, message):
    state['message_box'].config(state="normal")
    state['message_box'].insert('end', message + "\n" + "\n")
    state['message_box'].config(state="disabled")
    state['message_box'].see("end")  # Scroll to the bottom of the text box
    state['message_box'].update()  # Force an update of the widget
    # state['root'].update_idletasks()  # Process all pending events


def initialize_angle_lists(state):
    inc = np.float64(state['angle_inc_entry'].get())
    beta = np.radians(np.arange(0, 181, inc, dtype=np.float64))
    theta = np.radians(np.arange(0, 181, inc, dtype=np.float64))
    alpha = np.radians(np.arange(0, 181, inc, dtype=np.float64))
    phi = np.radians(np.arange(0, 361, inc, dtype=np.float64))
    state['beta'] = beta
    state['theta'] = theta
    state['alpha'] = alpha
    state['phi'] = phi



def initialize_csa_list(state):
    """Initialize the list of dCSA values used based on values entered."""
    csa_min = np.float64(state['delta_csa_entry'].get()) - np.float64(state['csa_range_entry'].get())
    csa_max = np.float64(state['delta_csa_entry'].get()) + np.float64(state['csa_range_entry'].get()) + np.float64(state['csa_step_entry'].get())
    csaList = np.arange(csa_min, csa_max, np.float64(state['csa_step_entry'].get()))

    state['csaList'] = csaList

def initialize_imported_data(state):
    """Initialize the tmlist based on decay data."""
    state['tmlist'] = state['decay_data'].iloc[:, 0].values  # Extract tmlist from decay data
    state['nmrsignal'] = state['decay_data'].iloc[:, 1].values # Extract intensity values from decay data

def calculate_phase_matrix_vectorized(state, m, masrate, csaList, beta, theta, alpha, phi, index):
    csa = (2 / 3) * csaList[index] * np.float64(state['freq_31p_entry'].get()) * 2 * np.pi  # CSA computation

    # Pre-factor for phase calculation
    z = (m * csa * np.sqrt(2)) / (2 * np.pi * masrate)

    # Create meshgrid for all angles
    beta_grid, theta_grid, alpha_grid, phi_grid = np.meshgrid(beta, theta, alpha, phi, indexing='ij')

    # Calculate phase1 and phase2 arrays without loops
    phase1 = z * np.sin(alpha_grid) * np.sin(2 * beta_grid)
    phase2 = z * (
        (np.sin(2 * beta_grid) * np.sin(alpha_grid) * (1 - (np.sin(theta_grid) ** 2) * (1 + np.cos(phi_grid) ** 2))) -
        (np.sin(beta_grid) * np.cos(alpha_grid) * (np.sin(theta_grid) ** 2) * np.sin(2 * phi_grid)) +
        (np.cos(2 * beta_grid) * np.sin(alpha_grid) * np.cos(phi_grid) +
         np.cos(beta_grid) * np.cos(alpha_grid) * np.sin(phi_grid)) * np.sin(2 * theta_grid)
    )

    # Compute cosdelphase for each combination of angles
    cosdelphase = np.cos(np.abs(phase2 - phase1))
    
    # Average over the phi and alpha axes, ensuring the result is a 2D matrix (len(beta) x len(theta))
    phase_matrix = np.mean(cosdelphase, axis=(2, 3))

    # Read out the phase_matrix at experimental csa
    # if index == state['experimental_csa_index']:
    #     print("phase_matrix shape: ", phase_matrix.shape)
    #     print(phase_matrix[:5, :5])  # Print the first 5x5 section of phase_matrix

    #     # Create a DataFrame from phase_matrix and add beta as row index
    #     df = pd.DataFrame(phase_matrix, index=beta, columns=theta)

    #     # Insert 'Beta/Theta' as the column name for the index
    #     df.index.name = 'Beta/Theta'

    #     # Write to CSV
    #     df.to_csv('python_phase_matrix.csv')
    #     print("phase_matrix exported successfully to 'python_phase_matrix.csv'")

    # Ensure phasematrix is of type float64
    phase_matrix = phase_matrix.astype(np.float64)

    # Store phase matrix in the state for later access
    state['phase_matrix'] = phase_matrix

    # Return the phase matrix along with the beta and theta for any necessary plotting or export
    return phase_matrix, beta, theta


@njit
def calculate_unptheta_vectorized(tau, theta):
    n_theta = len(theta)
    unptheta = np.zeros(n_theta, dtype=np.float64)
    if tau != 0.0:
        for k in range(n_theta):
            sqrt_term = np.sqrt(theta[k] * np.sin(theta[k]))
            exp_term = np.exp(-theta[k] ** 2 / (2.0 * tau))
            unptheta[k] = (1.0 / tau) * sqrt_term * exp_term
    else:
        for k in range(n_theta):
            if theta[k] == 0.0:
                unptheta[k] = 1.0
            else:
                unptheta[k] = 0.0  # Or handle appropriately
    return unptheta


@njit
def calculate_inner_loop_vectorized(tau, beta, theta, phase_matrix):
    n_beta = len(beta)
    n_theta = len(theta)
    if tau != 0.0:
        unptheta = calculate_unptheta_vectorized(tau, theta)
        sum_unptheta = 0.0
        for k in range(n_theta):
            sum_unptheta += unptheta[k]
        if sum_unptheta == 0.0:
            sum_unptheta = 1e-10  # Avoid division by zero
        ptheta = np.zeros(n_theta, dtype=np.float64)
        for k in range(n_theta):
            ptheta[k] = unptheta[k] / sum_unptheta
        
        ununrcodexsignaldecay = 0.0
        for i in range(n_beta):
            sin_beta_i = np.sin(beta[i])
            for j in range(n_theta):
                pbtmatrix = sin_beta_i * ptheta[j]
                codexmatrix = phase_matrix[i, j] * pbtmatrix
                ununrcodexsignaldecay += codexmatrix
    else:
        ununrcodexsignaldecay = 0.0
    return ununrcodexsignaldecay



def data_simulation(state, Dlat, m, beta, theta):
    try:
        # Convert variables to NumPy arrays of float64
        tmlist = np.ascontiguousarray(state['tmlist'], dtype=np.float64)
        nlipidprlist = np.ascontiguousarray(state['lsd_data'].iloc[:, 1].values, dtype=np.float64)
        radiuslist = np.ascontiguousarray(state['lsd_data'].iloc[:, 0].values, dtype=np.float64)
        phase_matrix = np.ascontiguousarray(state['phase_matrix'], dtype=np.float64)
        tr = np.float64(1 / float(state['mas_rate_entry'].get()))
        temp = np.float64(float(state['temp_entry'].get()))
        viscosity = np.float64(float(state['viscosity_entry'].get()))
        beta = np.ascontiguousarray(beta, dtype=np.float64)
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        m = np.float64(m)

        # Call the optimized simulation function
        simulated_codexsignaldecay = data_simulation_optimized(
            Dlat, tmlist, nlipidprlist, radiuslist, beta, theta, phase_matrix, tr, m, temp, viscosity
        )

        # Store the result in the state
        state['simulated_codexsignaldecay'] = simulated_codexsignaldecay
    except Exception as e:
        state['message_queue'].put(f"Data Simulation Error: {e}")
        raise

@njit(parallel=True)
def data_simulation_optimized(Dlat, tmlist, nlipidprlist, radiuslist, beta, theta, phase_matrix, tr, m, temp, viscosity):
    techo = tr * (m + 1)
    n_tmlist = tmlist.shape[0]
    n_radiuslist = radiuslist.shape[0]
    
    # Initialize arrays
    tau = np.zeros((n_tmlist, n_radiuslist), dtype=np.float64)
    ununrcodexsignaldecay = np.zeros((n_radiuslist, n_tmlist), dtype=np.float64)
    recouplingDecayWeighting = np.zeros(n_radiuslist, dtype=np.float64)
    
    # Precompute denominator for recouplingDecayWeighting
    denominator = np.zeros(n_radiuslist, dtype=np.float64)
    for j in range(n_radiuslist):
        denominator[j] = (4.0 * np.pi * viscosity * (radiuslist[j] * 1e-9) ** 3)
    
    # Compute recouplingDecayWeighting and tau
    for j in prange(n_radiuslist):
        term1 = (3.0 * temp * 1.38e-23) / denominator[j]
        term2 = (6.0 * Dlat) / ((radiuslist[j] * 1e-9) ** 2)
        recouplingDecayWeighting[j] = np.exp(-techo * (term1 + term2))
        
        for i in range(n_tmlist):
            tau[i, j] = (2.0 * Dlat * tmlist[i]) / ((1e-9 * radiuslist[j]) ** 2)
            
            # Compute ununrcodexsignaldecay
            ununrcodexsignaldecay[j, i] = calculate_inner_loop_vectorized(
                tau[i, j], beta, theta, phase_matrix
            )
    
    # Compute max_ununr
    max_ununr = np.zeros(n_radiuslist, dtype=np.float64)
    for j in range(n_radiuslist):
        max_val = ununrcodexsignaldecay[j, 0]
        for i in range(1, n_tmlist):
            if ununrcodexsignaldecay[j, i] > max_val:
                max_val = ununrcodexsignaldecay[j, i]
        if max_val == 0.0:
            max_val = 1e-10
        max_ununr[j] = max_val
    
    # Compute unrcodexsignaldecay
    unrcodexsignaldecay = np.zeros((n_radiuslist, n_tmlist), dtype=np.float64)
    for j in range(n_radiuslist):
        for i in range(n_tmlist):
            unrcodexsignaldecay[j, i] = (
                recouplingDecayWeighting[j] * nlipidprlist[j] *
                (ununrcodexsignaldecay[j, i] / max_ununr[j])
            )
    
    # Compute rcodexsignaldecay
    rcodexsignaldecay = np.zeros(n_tmlist, dtype=np.float64)
    for i in range(n_tmlist):
        sum_val = 0.0
        for j in range(n_radiuslist):
            sum_val += unrcodexsignaldecay[j, i]
        rcodexsignaldecay[i] = sum_val
    
    # Compute simulated_codexsignaldecay
    max_rcodexsignaldecay = rcodexsignaldecay[0]
    for val in rcodexsignaldecay[1:]:
        if val > max_rcodexsignaldecay:
            max_rcodexsignaldecay = val
    if max_rcodexsignaldecay == 0.0:
        max_rcodexsignaldecay = 1e-10  # Avoid division by zero
    
    simulated_codexsignaldecay = np.zeros(n_tmlist, dtype=np.float64)
    for i in range(n_tmlist):
        simulated_codexsignaldecay[i] = rcodexsignaldecay[i] / max_rcodexsignaldecay
    
    return simulated_codexsignaldecay

def objective_function(Dlat_array, state, m, beta, theta):
    # Extract scalar Dlat from the array
    Dlat = Dlat_array[0]
    
    # Extract necessary variables
    tmlist = state['tmlist']
    nlipidprlist = state['lsd_data'].iloc[:, 1].values
    radiuslist = state['lsd_data'].iloc[:, 0].values
    phase_matrix = state['phase_matrix']
    tr = 1 / np.float64(state['mas_rate_entry'].get())
    temp = np.float64(state['temp_entry'].get())
    viscosity = np.float64(state['viscosity_entry'].get())
    
    # Call the optimized simulation function
    simulated_codexsignaldecay = data_simulation_optimized(
        Dlat, tmlist, nlipidprlist, radiuslist, beta, theta,
        phase_matrix, tr, m, temp, viscosity
    )
    
    # Calculate residuals
    nmrsignal = state['nmrsignal']
    residuals = simulated_codexsignaldecay - nmrsignal
    sres = np.sum(residuals ** 2)
    
    return sres



def dlat_optimization(state, Dlat_init, m, beta, theta):
    
    # Perform optimization using scipy.optimize.minimize
    result = minimize(objective_function, Dlat_init, args=(state, m, beta, theta), method='Nelder-Mead')
    
    # Get the optimized Dlat value and the minimized sres
    Dlat_optimized = result.x[0]
    sres_minimized = result.fun
    
    return Dlat_optimized, sres_minimized


def calculate_dlat_error_bars_with_csa_range(state, m, beta, theta, sresListForCSAPlot, DlatListForCSAPlot):
    """
    Calculate Dlat error bars considering both the Hessian error and the variation 
    in Dlat over CSA values within 10% of the sum of squared residuals (sres).
    The function uses interpolated Dlat and sres values over the CSA range.
    """
    # Parameters
    dataRange = 0.1  # 10% tolerance for sres range
    epsilon = 0.01 * state['Dlat_before_error']  # Small perturbation factor of 1% for Hessian calculation
    csaList = state['csaList']

    # Get the index, Dlat, and sres associated with the experimental csa value
    experimental_csa_index = state['experimental_csa_index']
    Dlat_expt = DlatListForCSAPlot[experimental_csa_index]
    sres_expt = sresListForCSAPlot[experimental_csa_index]
    
    # Number of points for interpolation
    interpolatingPoints = 100 * len(csaList)  # 100 times the number of CSA points

    # Create fine CSA list for interpolation
    csaListFine = np.linspace(np.min(csaList), np.max(csaList), interpolatingPoints)

    # Interpolation for sres and Dlat, cubic by default but lower order if csaList values <3
    sresSpline = UnivariateSpline(csaList, sresListForCSAPlot, s=0, k=min(3, len(csaList)-1))
    DlatSpline = UnivariateSpline(csaList, DlatListForCSAPlot, s=0, k=min(3, len(csaList)-1))

    # Interpolated values over fine CSA range
    sresInterp = sresSpline(csaListFine)
    DlatInterp = DlatSpline(csaListFine)

    # Hessian-based error calculation (numerical second derivative)
    DlatHessianList = np.array([Dlat_expt + epsilon, Dlat_expt - epsilon])
    sresHessianList = np.zeros(2)
    
    # Recalculate sres values for small perturbations in Dlat
    for idx, Dlat in enumerate(DlatHessianList):
        # Simulate data for perturbed Dlat values
        data_simulation(state, Dlat, m, beta, theta)
        codexsignaldecay = state['simulated_codexsignaldecay']
        
        # Calculate sum of squared residuals (sres)
        residuals = codexsignaldecay - state['nmrsignal']
        sresHessianList[idx] = np.sum(residuals ** 2)

    # Hessian calculation (second derivative approximation)
    hessian = (sresHessianList[0] - 2 * sres_expt + sresHessianList[1]) / (epsilon ** 2)

    # Residual standard deviation
    df = len(state['tmlist']) - 1  # Degrees of freedom
    sigma = np.sqrt(sres_expt / df)

    # Standard error from Hessian
    varDlat_hessian = sigma ** 2 / hessian
    seDlat_hessian = np.sqrt(varDlat_hessian)

    # Define the acceptable sres range (±10% of sres_expt)
    sres_lower_bound = sres_expt * (1 - dataRange/2)
    sres_upper_bound = sres_expt * (1 + dataRange/2)
    
    # Initialize list to collect Dlat values within the sres range
    DlatRangeList = []

    searchStartIndex = len(csaListFine) // 2

    # Search for and collect the Dlat values within a window of 10% of the experimental sum of squared residuals

    # Positive Directional Search
    for j in range(len(csaListFine)):
        current_index = searchStartIndex + j
        if current_index >= len(csaListFine):
            break  # Prevent index out of bounds
        current_sres = sresInterp[current_index]
        if sres_lower_bound < current_sres < sres_upper_bound:
            DlatRangeList.append(DlatInterp[current_index])
        else:
            break  # Exit loop if condition not met
    
    # Negative Directional Search
    for j in range(len(csaListFine)):
        current_index = searchStartIndex - j
        if current_index < 0:
            break  # Prevent index out of bounds
        current_sres = sresInterp[current_index]
        if sres_lower_bound < current_sres < sres_upper_bound:
            DlatRangeList.append(DlatInterp[current_index])
        else:
            break  # Exit loop if condition not met
    
    DlatRangeArray = np.array(DlatRangeList)

    std_dev_csa_Dlat = np.std(DlatRangeArray)
    avDlat = np.mean(DlatRangeArray)

    # Propagate both Hessian and CSA range errors
    total_error = np.sqrt(seDlat_hessian ** 2 + std_dev_csa_Dlat ** 2)

    # Confidence interval using t-distribution (two-tailed, 95% confidence)
    alpha = 0.05
    t_value = t.ppf(1 - alpha/2, df)
    confidence_interval = t_value * total_error

    Dlat_before_error = state['Dlat_before_error'] 

     # Collect messages
    messages = []
    messages.append(f"Dlat prior to error calculation: {Dlat_before_error:.6e}")
    messages.append(f"Dlat after error calculation: {avDlat:.6e} with confidence interval: ±{confidence_interval:.4e}")

    state['message_queue'].put(f"Dlat prior to error calculation: {Dlat_before_error:.6e}")
    state['message_queue'].put(f"Dlat after error calculation: {avDlat:.6e} with confidence interval: ±{confidence_interval:.4e}")




def update_sres_plot(state, csaList, sresList):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    plt.rcParams['font.family'] = 'Arial'

    # Check if the canvas already exists
    if 'sres_canvas' in state:
        # Clear the existing figure
        state['sres_ax'].clear()
        ax = state['sres_ax']
    else:
        # Create a new figure and canvas
        fig, ax = plt.subplots(figsize=(3.0, 2.3))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        state['sres_canvas'] = FigureCanvasTkAgg(fig, master=state['sres_placeholder_canvas'])
        state['sres_canvas'].get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        state['sres_ax'] = ax

    ax.plot(csaList, sresList, marker='o', linestyle='-')
    ax.set_xlabel('δCSA (ppm)')
    ax.set_ylabel('Sum of Squared Residuals (sres)')
    ax.set_title('sres vs. δCSA')

    state['sres_canvas'].draw()

def update_dlat_plot(state, csaList, DlatList):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    plt.rcParams['font.family'] = 'Arial'

    # Check if the canvas already exists
    if 'dlat_canvas' in state:
        # Clear the existing figure
        state['dlat_ax'].clear()
        ax = state['dlat_ax']
    else:
        # Create a new figure and canvas
        fig, ax = plt.subplots(figsize=(3.0, 2.3))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        state['dlat_canvas'] = FigureCanvasTkAgg(fig, master=state['dlist_placeholder_canvas'])
        state['dlat_canvas'].get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        state['dlat_ax'] = ax

    ax.plot(csaList, DlatList, marker='o', linestyle='-')
    ax.set_xlabel('δCSA (ppm)')
    ax.set_ylabel('Dlat (m²/s)')
    ax.set_title('Dlat vs. δCSA')

    state['dlat_canvas'].draw()



def fitting_loop(state):
    state['message_queue'].put('Starting CODEX Curve Fitting...')

    # Extract inputs from GUI
    inputs = extract_gui_inputs(state)
    if inputs is None:
        # Simulation aborted due to input error
        state['message_queue'].put('Simulation aborted due to input error.')
        return None

    # Update state with extracted inputs for use in other functions
    state.update(inputs)

    # Initialization of input variables
    m = inputs['pulses_180']
    masrate = inputs['mas_rate']
    Dlat_init = inputs['dlat_guess'] * 1e-12

    # Initialize CSA list and angle lists
    initialize_csa_list(state)
    initialize_angle_lists(state)
    initialize_imported_data(state)

    csaList = state['csaList']
    beta = state['beta']
    theta = state['theta']
    tmlist = state['tmlist']  # Mixing times from imported data

    # Initialize arrays to store results
    sresListForCSAPlot = np.zeros(len(csaList))
    DlatListForCSAPlot = np.zeros(len(csaList))
    codexsignaldecaylist = np.zeros((len(csaList), len(tmlist)))

    state['experimental_csa_index'] = int(len(csaList) / 2)  # Index corresponding to experimental CSA
    experimental_csa_index = state['experimental_csa_index']

    for i in range(len(csaList)):
        # Check if the stop event is set
        if state['simulation_stop_event'].is_set():
            state['message_queue'].put('Simulation stopped by user.')
            return None

        # Calculate the phase matrix
        calculate_phase_matrix_vectorized(state, m, masrate, csaList, beta, theta, state['alpha'], state['phi'], i)

        # Optimize Dlat using scipy.optimize.minimize
        Dlat_optimized, sres = dlat_optimization(state, Dlat_init, m, beta, theta)

        # Compute and store the simulated CODEX signal decay
        data_simulation(state, Dlat_optimized, m, beta, theta)

        # Store the results for this CSA value
        sresListForCSAPlot[i] = sres
        DlatListForCSAPlot[i] = Dlat_optimized
        codexsignaldecaylist[i, :] = state['simulated_codexsignaldecay']

        # Print loop progress
        loopCounterText = 'CSA loop {}/{} completed for CSA = {:.1f} ppm giving Dlat = {:.5e}, sres = {:.5e}'.format(
            i + 1, len(csaList), csaList[i], Dlat_optimized, sres
        )
        state['message_queue'].put(loopCounterText)

    # Move this line outside the loop
    state['Dlat_before_error'] = DlatListForCSAPlot[experimental_csa_index]

    # Collect final message
    state['message_queue'].put('CODEX Curve Fitting Completed.')

    # Prepare results to return
    results = {
        'csaList': csaList,
        'sresListForCSAPlot': sresListForCSAPlot,
        'DlatListForCSAPlot': DlatListForCSAPlot,
        'codexsignaldecaylist': codexsignaldecaylist,
        'tmlist': tmlist,
        'experimental_csa_index': experimental_csa_index
    }

    return results



def update_gui_with_results(state, results):
    # Extract results
    csaList = results['csaList']
    sresListForCSAPlot = results['sresListForCSAPlot']
    DlatListForCSAPlot = results['DlatListForCSAPlot']
    codexsignaldecaylist = results['codexsignaldecaylist']
    tmlist = results['tmlist']
    experimental_csa_index = results['experimental_csa_index']

    # Remove previous fitted line if it exists
    if state.get('fitted_line') is not None:
        state['fitted_line'].remove()
        state['fitted_line'] = None

    # Update the CODEX decay curve
    fitted_line, = state['codex_ax'].plot([], [], 'r-', label='Fitted')
    fitted_line.set_data(tmlist, codexsignaldecaylist[experimental_csa_index, :])
    state['fitted_line'] = fitted_line
    state['codex_canvas'].draw()

    # Update sres vs. δCSA plot
    update_sres_plot(state, csaList, sresListForCSAPlot)

    # Update Dlat vs. δCSA plot
    update_dlat_plot(state, csaList, DlatListForCSAPlot)

    # Set Dlat_before_error before calculating error bars
    state['Dlat_before_error'] = DlatListForCSAPlot[experimental_csa_index]

    # Call the function to calculate Dlat error bars
    calculate_dlat_error_bars_with_csa_range(
        state,
        state['pulses_180'],
        state['beta'],
        state['theta'],
        sresListForCSAPlot,
        DlatListForCSAPlot
    )

