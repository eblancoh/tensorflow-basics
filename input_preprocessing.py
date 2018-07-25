########################################################################
# Functions and classes for processing images from the game-environment
# and converting them into a state.

def _rgb_to_black_and_white(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    # Set image to black and white (only 0's and 1's)
    # for the sake of simplicity
    img_gray[img_gray > 1] = 1

    img_gray = img_gray.astype(np.uint8)

    return img_gray

def _pre_process_image (image):
    """
    Convert an RGB-image into black and white image
    additionally, the image is cropped only to contain the most
    important part of the game. We are not interested in the edges
    of the screen.
    """

    # The image is set to black and white according to
    # _rgb_to_black_and_white function
    img = _rgb_to_black_and_white(image)

    # The image is cropped into 84x72 frame
    img = img[32:-10, 8:-8]
    img = img[::2, ::2]

    return img


class Input_to_Model:
    def __init__(self, image):
        """
        """

        # Pre-process the image and save it for later use.
        # The input image may be 8-bit integers but internally
        # we need to use floating-point to avoid image-noise
        # caused by recurrent rounding-errors.
        img = _pre_process_image(image=image)
        self.last_input = img.astype(np.float)

        # Set the last output to zero.
        # self.last_output = np.zeros_like(img)
        self.last_output = np.stack(tuple(self.last_input for _ in range(state_channels)), axis=2)

    def process(self, image):
        """Process a raw image-frame from the game-environment."""

        # Pre-process the image so it is gray-scale and resized.
        img = _pre_process_image(image=image)

        # Copy the contents of the input-image to the last input.
        self.last_input[:] = img[:]

        current_state = np.append(self.last_output[:, :, 1::],
                                  self.last_input.reshape([state_height, state_width, 1]), axis=2)

        self.last_output = current_state

        return current_state

    def get_state(self):

        state = self.last_output

        # Convert state to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state