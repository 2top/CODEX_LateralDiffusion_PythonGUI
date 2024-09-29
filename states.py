import tkinter as tk
from tkmacosx import Button
from functools import partial
from functions import select_codex_file, import_decay_data, select_lsd_file, plot_lsd_curve, run_simulation_in_thread, stop_simulation, export_codex_canvas, process_message_queue  
from queue import Queue


def init_vars(state):
    state['simulation_stop_event'] = None
    state['fitted_line'] = None 
    state['codex_file'] = None
    state['lsd_file'] = None
    state['delta_csa_entry_default'] = 20
    state['mas_rate_entry_default'] = 4000
    state['pulses_180_entry_default'] = 3
    state['freq_31p_entry_default'] = 162.0052
    state['temp_entry_default'] = 328.15
    state['viscosity_entry_default'] = 0.00055
    state['dlat_guess_entry_default'] = 1.0
    state['angle_inc_entry_default'] = 4
    state['csa_range_entry_default'] = 5
    state['csa_step_entry_default'] = 0.5

# Initialize a queue for messages
    state['message_queue'] = Queue()

def init_gui(root, state):

    # DATA IMPORT FRAME
    state['data_import_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Data Import")
    state['data_import_frame'].grid(row=0, column=0, padx=5, pady=5, sticky="nsew", rowspan=3, columnspan=2)

    # Configure grid weights for data_import_frame
    state['data_import_frame'].grid_columnconfigure(0, weight=1)  # First column absorbs extra space
    state['data_import_frame'].grid_columnconfigure(1, weight=0)  # Second column remains fixed

    # Widgets

    # CODEX File Selection

    state['select_codex_file_button'] = Button(state['data_import_frame'], text="Select File for CODEX Decay Signal", bg="#24a0ed", command=partial(select_codex_file, state))
    state['select_codex_file_button'].grid(row=0, column=0, padx=10, pady=10, sticky="w")

    state['codex_file_label'] = tk.Label(state['data_import_frame'], text="CODEX File:")
    state['codex_file_label'].grid(row=1, column=0, padx=5, pady=5, sticky="w")

    state['import_decay_data_button'] = Button(state['data_import_frame'], text="Import Decay Data", bg="#24a0ed", height=50, width=200, command=partial(import_decay_data, state))
    state['import_decay_data_button'].grid(row=0, column=1, padx=5, pady=5, rowspan=2,sticky="e")

    # Liposome Size Distribution File Selection
    state['select_lsd_file_button'] = Button(state['data_import_frame'], text="Select File for Liposome Size Distribution", bg="#24a0ed", command=partial(select_lsd_file, state))
    state['select_lsd_file_button'].grid(row=2, column=0, padx=10, pady=10, sticky="w")

    state['lsd_file_label'] = tk.Label(state['data_import_frame'], text="Distribution File:")
    state['lsd_file_label'].grid(row=3, column=0, padx=5, pady=5, sticky="w")

    state['import_distribution_button'] = Button(state['data_import_frame'], text="Import Distribution Data", bg="#24a0ed", height=50, width=200, command=partial(plot_lsd_curve, state))
    state['import_distribution_button'].grid(row=2, column=1, padx=5, pady=5, rowspan=2, sticky="e")





    # EXPERIMENTAL PARAMETERS FRAME
    state['expt_params_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Experimental Parameters")
    state['expt_params_frame'].grid(row=3, column=0, padx=5, pady=5, sticky="nsew", rowspan=5, columnspan=1)

    #Widgets

    # δCSA (ppm)
    state['delta_csa_label'] = tk.Label(state['expt_params_frame'], text="δCSA (ppm):")
    state['delta_csa_label'].grid(row=0, column=0, padx=5, pady=5, sticky="w")

    state['delta_csa_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['delta_csa_entry'].grid(row=0, column=1, padx=5, pady=5, sticky="w")
    state['delta_csa_entry'].insert(0, state['delta_csa_entry_default'])

    # MAS Rate (Hz)
    state['mas_rate_label'] = tk.Label(state['expt_params_frame'], text="MAS Rate (Hz):")
    state['mas_rate_label'].grid(row=1, column=0, padx=5, pady=5, sticky="w")

    state['mas_rate_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['mas_rate_entry'].grid(row=1, column=1, padx=5, pady=5, sticky="w")
    state['mas_rate_entry'].insert(0, state['mas_rate_entry_default'])

    # 180 pulses
    state['pulses_180_label'] = tk.Label(state['expt_params_frame'], text="Number of 180 pulses:")
    state['pulses_180_label'].grid(row=2, column=0, padx=5, pady=5, sticky="w")

    state['pulses_180_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['pulses_180_entry'].grid(row=2, column=1, padx=5, pady=5, sticky="w")
    state['pulses_180_entry'].insert(0, state['pulses_180_entry_default'])

    # 31P freq. (MHz)
    state['freq_31p_label'] = tk.Label(state['expt_params_frame'], text="31P freq. (MHz):")
    state['freq_31p_label'].grid(row=3, column=0, padx=5, pady=5, sticky="w")

    state['freq_31p_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['freq_31p_entry'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
    state['freq_31p_entry'].insert(0, state['freq_31p_entry_default'])

    # temperature (K)
    state['temp_label'] = tk.Label(state['expt_params_frame'], text="Temperature (K):")
    state['temp_label'].grid(row=4, column=0, padx=5, pady=5, sticky="w")

    state['temp_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['temp_entry'].grid(row=4, column=1, padx=5, pady=5, sticky="w")
    state['temp_entry'].insert(0, state['temp_entry_default'])

    # viscosity (Pa s)
    state['viscosity_label'] = tk.Label(state['expt_params_frame'], text="Viscosity (Pa s):")
    state['viscosity_label'].grid(row=5, column=0, padx=5, pady=5, sticky="w")

    state['viscosity_entry'] = tk.Entry(state['expt_params_frame'], width=8)
    state['viscosity_entry'].grid(row=5, column=1, padx=5, pady=5, sticky="w")
    state['viscosity_entry'].insert(0, state['viscosity_entry_default'])

    



    # SIMULATION/FITTING PARAMETERS FRAME
    state['sim_params_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Simulation Parameters")
    state['sim_params_frame'].grid(row=8, column=0, padx=5, pady=5, sticky="nsew", rowspan=2, columnspan=2)

    # Widgets

    # Dlat guess (in m^2 s^-1)
    state['dlat_guess_label'] = tk.Label(state['sim_params_frame'], text="Dlat guess (10^-12 m^2 s^-1):")
    state['dlat_guess_label'].grid(row=0, column=0, padx=5, pady=5, sticky="w")

    state['dlat_guess_entry'] = tk.Entry(state['sim_params_frame'], width=8)
    state['dlat_guess_entry'].grid(row=0, column=1, padx=5, pady=5, sticky="w")
    state['dlat_guess_entry'].insert(0, state['dlat_guess_entry_default'])

    # angle increment (deg)
    state['angle_inc_label'] = tk.Label(state['sim_params_frame'], text="Angle increment (deg):")
    state['angle_inc_label'].grid(row=1, column=0, padx=5, pady=5, sticky="w")

    state['angle_inc_entry'] = tk.Entry(state['sim_params_frame'], width=8)
    state['angle_inc_entry'].grid(row=1, column=1, padx=5, pady=5, sticky="w")
    state['angle_inc_entry'].insert(0, state['angle_inc_entry_default'])

    # CSA Range
    state['csa_range_label'] = tk.Label(state['sim_params_frame'], text="δCSA Range (± value, ppm):")
    state['csa_range_label'].grid(row=0, column=2, padx=5, pady=5, sticky="w")

    state['csa_range_entry'] = tk.Entry(state['sim_params_frame'], width=8)
    state['csa_range_entry'].grid(row=0, column=3, padx=5, pady=5, sticky="w")
    state['csa_range_entry'].insert(0, state['csa_range_entry_default'])

    # CSA Step
    state['csa_step_label'] = tk.Label(state['sim_params_frame'], text="δCSA Step Increment (ppm):")
    state['csa_step_label'].grid(row=1, column=2, padx=5, pady=5, sticky="w")

    state['csa_step_entry'] = tk.Entry(state['sim_params_frame'], width=8)
    state['csa_step_entry'].grid(row=1, column=3, padx=5, pady=5, sticky="w")
    state['csa_step_entry'].insert(0, state['csa_step_entry_default'])



    # MESSAGE BOX FRAME
    state['message_box_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Message Box")
    state['message_box_frame'].grid(row=10, column=0, padx=5, pady=5, sticky="nsew", rowspan=2, columnspan=5)

    # Message box
    state['message_box'] = tk.Text(state['message_box_frame'], height=10, width=150, state='disabled', font=("Arial", 11))
    state['message_box'].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # # Clear message box button
    # state['clear_message_box_button'] = Button(state['message_box_frame'], text="Clear Message Box", bg='#24a0ed', height=35, width=10, command=partial(clear_message_box, state))
    # state['clear_message_box_button'].grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    # Create a scrollbar
    scrollbar = tk.Scrollbar(state['message_box_frame'])
    scrollbar.grid(row=0, column=1, sticky='ns')

    # Configure the message box to work with the scrollbar
    state['message_box'].config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=state['message_box'].yview)



    # CODEX CANVAS FRAME
    state['codex_canvas_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "CODEX Plot")
    state['codex_canvas_frame'].grid(row=0, column=2, padx=5, pady=5, sticky="nsew", rowspan=6 ,columnspan=3)

    # Fit Curve button
    state['fit_codex_curve_button'] = Button(state['codex_canvas_frame'], text="Fit CODEX Curve", bg="#24a0ed", height=50, width=200, command=partial(run_simulation_in_thread, state))
    state['fit_codex_curve_button'].grid(row=0, column=0, padx=5, pady=5, sticky="nesw")

    # Stop Simulation button
    state['stop_simulation_button'] = Button(state['codex_canvas_frame'], text="Stop Simulation", bg="#24a0ed", height=50, width=200, command=partial(stop_simulation, state))
    state['stop_simulation_button'].grid(row=1, column=0, padx=5, pady=5, sticky="nesw")
    state['stop_simulation_button']['state'] = 'disabled'  # Disable initially

    # Export Figure button
    state['export_figure_button'] = Button(state['codex_canvas_frame'], text="Export Figure", bg="#24a0ed", height=50, width=200, command=partial(export_codex_canvas, state))   
    state['export_figure_button'].grid(row=2, column=0, padx=5, pady=5, sticky="nesw")

    # Placeholder canvas
    state['codex_placeholder_canvas'] = tk.Canvas(state['codex_canvas_frame'], width=400, height=300, background="white", bd=2, relief="solid")
    state['codex_placeholder_canvas'].grid(row=0, column=1, padx=5, pady=5, sticky="nesw", rowspan=3)


    # LSD CANVAS FRAME

    state['lsd_canvas_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Liposome Size Distribution")
    state['lsd_canvas_frame'].grid(row=3, column=1, padx=5, pady=5, sticky="nsew", rowspan=5, columnspan=1)

    state['lsd_canvas_frame'].grid_rowconfigure(1, weight=1)

    state['lsd_placeholder_canvas'] = tk.Canvas(state['lsd_canvas_frame'], width=300, height=230, background="white", bd=2, relief="solid")
    state['lsd_placeholder_canvas'].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")


    # OTHER PLOTS FRAME
    state['other_plots_frame'] = tk.LabelFrame(root, padx=5, pady=5, text = "Other Plots")
    state['other_plots_frame'].grid(row=6, column=2, padx=5, pady=5, sticky="nsew", rowspan=4, columnspan=3)


    # sres Placeholder canvas
    state['sres_placeholder_canvas'] = tk.Canvas(state['other_plots_frame'], width=300, height=230, background="white", bd=2, relief="solid")
    state['sres_placeholder_canvas'].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # Dlist vs csaList canvas
    state['dlist_placeholder_canvas'] = tk.Canvas(state['other_plots_frame'], width=300, height=230, background="white", bd=2, relief="solid")
    state['dlist_placeholder_canvas'].grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

    process_message_queue(state)
