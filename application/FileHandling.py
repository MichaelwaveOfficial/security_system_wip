from typing import List, Dict
from flask import request
import os

class FileHandling(object):

    '''
    FileHandling class to seperate multiple functions concerned with managing the captures kept within the devices local storage, bunlding them together in one location.
    '''

    def __init__(self, CAPTURES_DIRECTORY : str = './static/captures/', FORMATTED_FILENAME_DATE : str = '%a-%b-%Y_%I-%M-%S%p', FORMATTED_DISPLAY_DATE : str = '%I:%M:%S%p', MAXIMUM_FILES_STORED : int = 30) -> None:
        
        # Initialise list that will store the images metadata.
        self.stored_images : List = []

        # Store value of the file order when displaying captures stored on screen.
        self.file_order : bool = False

        # Directory to store captured frames. 
        self.CAPTURES_DIRECTORY = CAPTURES_DIRECTORY

        # Filename format containing d-m-y_m-h-s-am/pm.
        self.FORMATTED_FILENAME_DATE = FORMATTED_FILENAME_DATE

        # Time format for frame display.
        self.FORMATTED_DISPLAY_DATE = FORMATTED_DISPLAY_DATE

        # Final vairiable to control maximum number of files allowed within the devices local storage.
        self.MAXIMUM_FILES_STORED = MAXIMUM_FILES_STORED


    def sort_files(self, files : List[dict], reverse_order : bool = False) -> List[dict]:

        '''
        Sort list containing captures from newest to oldest from the order parameterised, OS natively sorts in order due to filenames date+time structure,
        meaning it is simple enough to reverse the list these entries are stored in. 

        :param: files - List of files to be sorted. 
        :param: reverse_order - Order required for files to be sorted into.
        :return: sorted_list - Sorted list of images.
        '''

        # Copy list, do NOT modify directly.
        sorted_list = files.copy()
        
        # If reverse order parameterised, reverse the list using .reverse() BIF.  
        if reverse_order:
            sorted_list.reverse()
        else:
            # Otherwise reutrn list in default order. 
            sorted_list.sort(
                key = lambda cap: cap['capture_date'] + cap['capture_time'],
            )

        # Return list of files in desired order. 
        return sorted_list
    

    def access_stored_captures(self, directory: str) -> List[Dict[str, str]]:

        '''
        Access images stored locally on the device. By iterating over each file, it will accumulate the meta data associated
        with each image, appending it to a dictionary, forming the metadata for an image which can be rendered into a html template,
        sanitised which can provide useful output for the user. 

        :params: directory - Access the cameras capture directory attribute.
        :return: stored_images - List consisting of dictionaries containing an images metadata for later access. 
        (img = {'fullpath','filename','file_ext', 'capture_date'})
        '''

        # Loop over all image files found within the captures directory. 
        for file in os.listdir(directory):

            # Check the file extensions match those desired.
            if file.endswith(('.jpg', '.jpeg', '.png')):

                # Append the filename to the captures directory path.
                fullpath = os.path.join(directory, file)
                # Get the standalone filename (date) and the file extention.
                filename, file_ext = os.path.splitext(file)
                # Access just the date.
                capture_date = filename.split('_')[0]
                # Access the time. 
                capture_time = filename.split('_')[1]

                # Append the images data to the images dictionary. 
                self.stored_images.append({
                    # Full filepath.
                    'fullpath': fullpath,
                    # Standalone filename.
                    'filename': filename,
                    # Files extension.
                    'file_ext': file_ext,
                    # Date it was captured.
                    'capture_date': capture_date, 
                    # Time the capture was taken.
                    'capture_time' : capture_time,
                })

        # Return dictionary containing meta data associated with images.
        return self.stored_images
    

    def manage_images_displayed(self, max_images : int, stored_images : Dict[str, str]) -> tuple[str, int, int]:

        '''
        Controls how many images are displayed onto a page, if that image is breached the overflow will be moved onto the next. 
        
        :param: max_images - Maximum number of images to fit onto a page. 
        :param: stored_images - List containing images first and last images to be sliced. 
        :return: current_images - Images to display based on the slice taken from the images array after calculating their indexes. 
        :return: total_pages - Pages required to fit all of the images gathered. 
        :return: page_number - Current page number to display to the user. 
        '''

        # Current page number index. 
        page_number = request.args.get('page', default=1, type=int)

        # Initial starting page to work with. 
        initial_page = (page_number - 1) * max_images

        # Final ending page. 
        final_page = initial_page + max_images

        # Indexes of images to be displayed upon the page.  
        current_images = stored_images[initial_page:final_page]

        # Calculate total number of pages to be traversed. 
        total_pages = len(stored_images) // max_images + (1 if len(stored_images) % max_images != 0 else 0)

        # return values for access later. 
        return current_images, total_pages, page_number
    
    
    def check_file_exhaustion(self, directory : str, file_limit : int) -> None:

        '''
        Check stored files, remove oldest captures in order to mitigate resource exhaustion, can be set by user. 

        :param: directory - Specified directory where files are stored. 
        :param: file_limit - Maximum number of files allowed within the devices local storage. 
        :return: N/A.
        '''
        
        # Source all files from the directory parameterised. 
        files = os.listdir(directory)

        # If the length of the files exceed the limit. 
        if len(files) > file_limit:
            
            # Iterate over files, starting at the index where the limit is set. 
            for file in files[file_limit:]:

                # Get files full path, joining the directory and filename.
                fullpath = os.path.join(directory, file)

                # Remove that file using the fullpath.
                os.remove(fullpath)

                # Notify users changes have been applied.
                print(f'Storage Limits Exceeded!\n {fullpath} has been deleted from the system!')
    