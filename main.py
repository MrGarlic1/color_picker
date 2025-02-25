import numpy as np
import cv2


def get_palette(img):
    """
    Return palette in descending order of frequency
    """
    arr = np.asarray(img)
    palette, index = np.unique(asvoid(arr).ravel(), return_inverse=True)
    palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
    count = np.bincount(index)
    order = np.argsort(count)
    return palette[order[::-1]]


def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def main():

    filename = "elephant.png"
    if filename.endswith(".png"):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        flat_img = img.reshape(-1, 4)
        flat_img = flat_img[flat_img[:, 3] > 200][:, :3]
    else:
        img = cv2.imread(filename)
        flat_img = img.reshape(-1, 3)

    avg_color = np.average(flat_img, axis=0).astype(int)[0:3]
    total_pixels = len(flat_img)

    color_group_range = 35
    primary_min_distance_from_average = 50
    primary_max_brightness = 300
    accent_min_distance_from_primary = 100
    accent_min_brightness = 1
    accent_min_distance_from_average = 20
    primary_freq_threshold = 0.05
    accent_freq_threshold = 0.005

    groupings = []

    for i in range(110):
        base_color = flat_img[0, :]

        condition = (
            (abs(flat_img[:, 0] - base_color[0]) < color_group_range/2) &
            (abs(flat_img[:, 1] - base_color[1]) < color_group_range / 2) &
            (abs(flat_img[:, 2] - base_color[2]) < color_group_range / 2)
            )

        groupings.append((base_color, len(flat_img[condition])))

        flat_img = flat_img[~condition]
        if len(flat_img) <= 1:
            break

    groupings = sorted(groupings, key=lambda x: x[1])
    print(groupings)


    for color in groupings:
        print(np.abs(color[0]).sum())
        print(color[0])

    primary_candidates = [
        color for color in groupings if
        sum(map(abs, color[0])) < primary_max_brightness and
        color[1] > total_pixels*primary_freq_threshold and
        sum(map(abs, color[0] - avg_color)) > primary_min_distance_from_average
    ]

    if not primary_candidates:
        primary_candidates = [
            color for color in groupings if
            sum(map(abs, color[0])) < primary_max_brightness and
            color[1] > total_pixels * primary_freq_threshold
        ]
    if not primary_candidates:
        primary_candidates = [
            color for color in groupings if
            sum(map(abs, color[0])) < primary_max_brightness
        ]

    if not primary_candidates:
        primary_candidates = groupings

    primary_color = primary_candidates[0][0]

    accent_candidates = [
        color for color in groupings if
        np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
        np.abs(color[0]).sum() > accent_min_brightness and
        color[1] > total_pixels * accent_freq_threshold and
        np.abs(color[0] - avg_color).sum() > accent_min_distance_from_average
    ]
    if not accent_candidates:
        accent_candidates = [
            color for color in groupings if
            np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
            np.abs(color[0]).sum() > accent_min_brightness and
            color[1] > total_pixels * accent_freq_threshold
        ]
    if not accent_candidates:
        accent_candidates = [
            color for color in groupings if
            np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
            np.abs(color[0]).sum() > accent_min_brightness
        ]
    print(accent_candidates)
    accent_color = accent_candidates[0][0]

    print(avg_color)
    print(primary_color)
    print(accent_color)

    ## TESTING/REVIEWING CODE

    avg_image = np.full(
        shape=(100, 100, 3),
        fill_value=avg_color,
        dtype=np.uint8
    )
    primary_image = np.full(
        shape=(100, 100, 3),
        fill_value=primary_color,
        dtype=np.uint8
    )
    accent_image = np.full(
        shape=(100, 100, 3),
        fill_value=accent_color,
        dtype=np.uint8
    )
    cv2.imshow("average", avg_image)
    cv2.imshow("original", img)
    cv2.imshow("primary", primary_image)
    cv2.imshow("accent", accent_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
