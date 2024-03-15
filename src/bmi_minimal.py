from numpy import ndarray
from bmipy import Bmi

from typing import Tuple

class Bmi_Minimal(Bmi):
    """Intermediate ABC for implementing functionality of standard BMI
       which isn't strictly needed for "functional" BMI.  This class
       simply implements the functions which raise NotImplementedError

       A not very useful, but fully functional BMI model likely only requires
       
       initialize()
       update()
       finalize()

       implementations.  Everything else builds upon this.  Once input/output variables
       are introduced then the variable metadata is required for technical exchange, e.g.

       get_var_itemsize()
       get_var_nbytes()

       This class provides a means for incrementally building BMI functionality
       by throwing runtime errors for functionality that is requested but not available
       compared to requring a BMI class to implement every abstract function even if it
       doesn't make sense for the model (e.g. unstructured mesh functions for a model with
       with only scalar values.)

       Exceptions to the behavoir above are
       get_component_name -- will return the subclass class name if not overridden
       get_value -- returns a call to get_value_pointer and copies data

    Args:
        Bmi (Bmi): Base BMI abstract class
    """
    #############
    # Bmi functions which have a reasonable "default" implementation
    #############

    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns:
            str: Model Name
        """
        return self.__class__.__name__

    # Some "optional" functions which have a reasonable default implementation
    # as long as get_value_ptr is implemented and tye typing is adhered to
    # TODO should these try/except to catch the get_value_ptr unimplmented error
    # and raise a more tailored exception indicating the problem/solution?

    def get_value(self, name: str, dest: ndarray) -> ndarray:
        dest[:] = self.get_value_ptr(name)
    
    def get_var_nbytes(self, name: str) -> int:
        """Get the number of total bytes required to represent the variable.

        Args:
            name (str): Name of variable.

        Returns:
            int: Size of data array in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of the variable.
        
           If the variable is an array, this is the type of a single
           element of the array.

        Args:
            name (str): Name of variable.

        Returns:
            str: Data type.
        """
        return str(self.get_value_ptr(name).dtype)

    ###############
    # BMI functions which may be cosidered "optional" for a minimally functioning
    # BMI implmentation
    ###############
    def update_until(self, time: float) -> None:
        """Update model from current_time until current_time + time

        Args:
            time (float): duration of time to advance model till
        """
        raise NotImplementedError()

    # BMI Variable Information Functions
    def get_input_item_count(self) -> int:
        """Number of model input variables

        Returns:
            int: number of input variables
        """
        raise NotImplementedError()
    
    def get_input_var_names(self) -> Tuple[str]:
        """The names of each input variables

        Returns:
            Tuple[str]: iterable tuple of input variable names
        """
        raise NotImplementedError()

    def get_output_item_count(self) -> int:
        """Number of model output variables

        Returns:
            int: number of output variables
        """
        raise NotImplementedError()
    
    def get_output_var_names(self) -> Tuple[str]:
        """The names of each output variable

        Returns:
            Tuple[str]: iterable tuple of output variable names
        """
        raise NotImplementedError()

    # BMI Variable Information Functions
    def get_var_grid(self, name: str) -> int:
        """Get the grid identiferier associated with a given variable

        Args:
            name (str): name of the variable

        Raises:
            UnknownBMIVariable: name is not recognized, grid unknown

        Returns:
            int: grid identifier associated with @p name
        """
        raise NotImplementedError()
    
    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns:
            int: number of bytes representing a single variable of @p name
        """
        raise NotImplementedError()
    
    def get_var_location(self, name: str) -> str:
        """Location of the variable relative to the grid

        Args:
            name (str): name of the BMI variable
        
        Raises:
            UnknownBMIVariable: name is not recognized, location unknown

        Returns:
            str: location on the grid, e.g. node, face
        """
        raise NotImplementedError()
    
    def get_var_units(self, name: str) -> str:
        """Get units of the given variable

        Args:
            name (str): variable name

        Raises:
            UnknownBMIVariable: name is not recognized, units unknown

        Returns:
            str: units
        """
        raise NotImplementedError()

    def get_var_grid(self, name: str) -> int:
        raise NotImplementedError()
    
    def get_var_location(self, name: str) -> str:
        raise NotImplementedError()
    
    def get_var_units(self, name: str) -> str:
        raise NotImplementedError()

    def get_current_time(self) -> float:
        raise NotImplementedError()
    
    def get_end_time(self) -> float:
        raise NotImplementedError()
    
    # BMI grid functions
    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError()
    
    def get_grid_edge_nodes(self, grid: int, edge_nodes: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError()
    
    def get_grid_face_edges(self, grid: int, face_edges: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_face_nodes(self, grid: int, face_nodes: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_node_count(self, grid: int) -> int:
        raise NotImplementedError()
    
    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_origin(self, grid: int, origin: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_rank(self, grid: int) -> int:
        raise NotImplementedError()
    
    def get_grid_shape(self, grid: int, shape: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_size(self, grid: int) -> int:
        raise NotImplementedError()
    
    def get_grid_spacing(self, grid: int, spacing: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_type(self, grid: int) -> str:
        raise NotImplementedError()
    
    def get_grid_x(self, grid: int, x: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_y(self, grid: int, y: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_grid_z(self, grid: int, z: ndarray) -> ndarray:
        raise NotImplementedError()
        
    def get_start_time(self) -> float:
        raise NotImplementedError()
    
    def get_time_step(self) -> float:
        raise NotImplementedError()
    
    def get_time_units(self) -> str:
        raise NotImplementedError()
    
    # BMI get/set
    def get_value_ptr(self, name: str) -> ndarray:
        raise NotImplementedError

    def get_value_at_indices(self, name: str, dest: ndarray, inds: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def set_value(self, name: str, src: ndarray) -> None:
        raise NotImplementedError()
    
    def set_value_at_indices(self, name: str, inds: ndarray, src: ndarray) -> None:
        raise NotImplementedError()
