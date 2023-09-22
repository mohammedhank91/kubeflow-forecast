import kfp
from kfp import components

# Load your custom Python functions
import main_shift_cycle
import main_simple_shift
import forecast
import load_data

    
# Define pipeline components using your custom functions
load_data_op = components.create_component_from_func(load_data.load_data)
forecast_op = components.create_component_from_func(forecast.forecast)
main_shift_cycle_op = components.create_component_from_func(main_shift_cycle.main_shift_cycle)
main_simple_shift_op = components.create_component_from_func(main_simple_shift.main_simple_shift)


@kfp.dsl.pipeline(name="My Forecasting Pipeline")
def forecasting_pipeline(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version):
    # Define pipeline steps using your components
    load_data_task = load_data_op()
    forecast_task = forecast_op(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
    main_shift_cycle_task = main_shift_cycle_op(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
    main_simple_shift_task = main_simple_shift_op(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
    

    # Define dependencies between tasks
    load_data_task
    forecast_task.after(load_data_task)
    main_shift_cycle_task.after(forecast_task)
    main_simple_shift_task.after(forecast_task)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(forecasting_pipeline, "forecasting_pipeline.yaml")
