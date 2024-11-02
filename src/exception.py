# Custom Exceptions Handling - The Goal is to create more detailed error messages, which tells us where in the code error happened(file name, line number) and what went wrong(error message).
import sys # Module allows some 'varibales and functions' to interact with Python-interpriter. Used to gather details about where an error occured.

def get_error_message(error, error_detail : sys): # To Generate detailed error message.
    
    _, _, exc_tb = error_detail.exc_info() # Returns details about error like file_name, line_no....etc
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    error_message = f"Error occured in {file_name} \n {exc_tb.tb_lineno} line-number \n error-message: {str(error)} \n"
    
    return error_message

# Customizes error message.
class CustomException(Exception):
    def __init__(self, error_message, error_detail : sys):
        super().__init__(error_message)
        
        self.error_message = get_error_message(error_message, error_detail = error_detail)
        
    def __str__(self): # Converts Object into String representation.
        return self.error_message
    
    # Without __str__
    # Prints something like: <0x7f8b3c1d2e40> - CustomException Object.
    # Using __str__, we can directly print an object and get readable output instead of seeing the object Memory-Location.
    