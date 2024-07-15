import cv2
import numpy as np

id = 0
points = []
id_polylines = {}
drawing = False

line_points = []


def parking_draw(event, x, y, flag, params):
    """
    Get the parking area's coordinates from the first frame of the video
    """
    global id, points, id_polylines, drawing
    frame = params["frame"]
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        id += 1
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        if id not in id_polylines:
            id_polylines[id] = []
        id_polylines[id].append(np.array(points, np.int32))
        cv2.polylines(img=frame,
                      pts=[np.array(points, np.int32)],
                      isClosed=True,
                      color=(0, 0, 255),
                      thickness=2)
        cv2.imshow("Frame for drawing coordinates", frame)


def line_draw(event, x, y, flag, params):
    global line_points, drawing
    frame = params["frame"]
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        line_points.append((x, y))
        drawing = False
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)
        # cv2.imshow("Frame for drawing coordinates", frame)


def get_parking_coordinates(source) -> dict:
    """
    Get the parking area's coordinates
    """
    cap = cv2.VideoCapture(source)
    _, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (1280, 720))
    draw_frame = first_frame.copy()
    cv2.namedWindow("Frame for drawing coordinates")
    cv2.setMouseCallback("Frame for drawing coordinates",
                         parking_draw, {"frame": draw_frame})

    # Given a list of coordinates, we can draw rectangle around all parking areas
    while True:
        cv2.imshow("Frame for drawing coordinates", draw_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

    return id_polylines


def get_line_coordinates(source) -> list:
    """
    Get the line coordinates
    """
    cap = cv2.VideoCapture(source)
    _, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (1280, 720))
    draw_frame = first_frame.copy()
    cv2.namedWindow("Frame for drawing line")
    cv2.setMouseCallback("Frame for drawing line",
                         line_draw, {"frame": draw_frame})

    while True:
        cv2.imshow("Frame for drawing line", draw_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

    return line_points


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (pos[0] - 5, pos[1] - 5), (x + text_w, y +
                  text_h + 5), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)),
                font, font_scale, text_color, font_thickness)


def direction(line_p1, line_p2, point):
    return (point[0] - line_p1[0]) * (line_p2[1] - line_p1[1]) - (point[1] - line_p1[1]) * (line_p2[0] - line_p1[0])


def is_crossed_line(line_coordinates, prev_center, current_center):
    crossed = False
    line_p1, line_p2 = line_coordinates
    d1 = direction(line_p1, line_p2, prev_center)
    d2 = direction(line_p1, line_p2, current_center)
    if (d1 > 0) != (d2 > 0):  # using perp dot product
        crossed = True
    return crossed


def get_direction(previous_point, current_point):
    """
    Get the direction of the moving vehicle
    """
    if current_point[1] >= previous_point[1]:
        return "in"
    elif current_point[1] < previous_point[1]:
        return "out"
