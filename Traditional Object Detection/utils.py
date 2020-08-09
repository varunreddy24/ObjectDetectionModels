import imutils

def sliding_window(image,step,ws):
    '''
    This function will help return ROI of an image passed in the argument

    Arguments:
        image: input of the image that we are going to extract ROI.
        step: It defines the number of times the image has to be looped over.
        ws: defines the window width at each step that will be looped over
    
    Returns:
        A generator is returned with following patameters:
            x: the x coordinate of the image
            y: the y coordinate of the image
            image: the cropped image
    '''
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            yield (x,y,image[y,y+ws[1],x:x+ws[0]])

def image_pyramid(image,scale=1.5,min_size=(100,100)):
    '''
    This function will return the image pyramid of a given image
    
    Arguments:
        image: The generated images will be returned
    '''
    yield image

    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image,width=w)

        if image.shape[1]<min_size[0] or image.shape[0]<min_size[0]:
            break

        yield image
    
