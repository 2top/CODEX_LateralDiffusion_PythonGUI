import tkinter as tk
from states import init_gui, init_vars
import tkinter.messagebox as messagebox

def on_closing(state):
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Signal the simulation thread to stop
        if 'simulation_stop_event' in state and state['simulation_stop_event'] is not None:
            state['simulation_stop_event'].set()

        # Cancel any scheduled 'after' calls
        if 'process_message_queue_after_id' in state:
            state['root'].after_cancel(state['process_message_queue_after_id'])

        # Quit the main loop
        state['root'].quit()
        # Destroy the root window
        state['root'].destroy()

def main():
    root = tk.Tk()
    state = {}
    state['root'] = root
    root.title("CODEX Decay Curve Fitting Algorithm")
    root.geometry("1260x880")
    root.attributes('-fullscreen', False)

    from states import init_gui, init_vars

    init_vars(state)
    init_gui(root, state)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(state))

    root.mainloop()

if __name__ == "__main__":
    main()
