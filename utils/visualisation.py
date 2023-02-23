import cv2

color_map = {
    0: [0, 0, 0],  # black
    1: [0, 0, 255],  # Blue
    2: [0, 255, 0],  # Green
    3: [255, 0, 0],  # Red
    4: [255, 255, 0],  # Yellow
}


def result_visualisation(img, idx, coordinates):
    """
    putting prediction onto the image
    input: image in RGB format
    idx: list of lane_idx e.g. [1,2,3]
    coordinates: list e.g. [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    # if there is no coordinates
    if len(coordinates) == 0:
        return img

    # if there are coordinates in the list
    for i in range(len(coordinates)):
        coords = coordinates[i]
        lane_idx = idx[i]
        colour = color_map[lane_idx]
        for j in range(len(coords)):
            if (j + 1) < len(coords):
                sp = coords[j]
                ep = coords[j + 1]
                if sp[0] < 0 or ep[0] < 0:
                    continue
                cv2.line(img, sp, ep, colour, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
