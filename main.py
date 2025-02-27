import numpy as np
import cv2
import logging


def read_image(filename: str) -> np.array:
    """
    Inputs: Relative/absolute path to image (string)
    Outputs: Array of pixel HLS color values, sorted by frequency (nx3 numpy array)
    Array of pixel RGB color values, sorted by frequency (nx3 numpy array)
    """
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    assert isinstance(img, np.ndarray), "Image not found. Check file path?"

    if max(img.shape) > 600:
        scale_factor = 600/max(img.shape)
        img = cv2.resize(
            img, (int(img.shape[0]*scale_factor), int(img.shape[1]*scale_factor)), interpolation=cv2.INTER_AREA
        )

    # Check if alpha channel is present, and if so, filter out background
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = img.reshape(-1, 4)
        img = img[img[:, 3] > 250][:, :3]
        # img = img[:, 0:3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(-1, 3)

    img_hls = cv2.cvtColor(np.array([img]), cv2.COLOR_RGB2HLS)
    img_hls = img_hls.reshape(-1, 3)

    return img, img_hls


def get_color_frequency(img: np.array, color_group_range: int):
    """
    Inputs: img - Image RGB color values (nx3 numpy array)
    color_group_range - width of color groups; a small value yields higher hue accuracy, lower freq. accuracy (int)
    Outputs: Image - Image unique color groupings (nx3 numpy array)
    Frequencies - Corresponding frequencies of each color group in the image (n numpy array)
    """
    rgb_vals = range(round(color_group_range // 2), int(255 + color_group_range / 1.5), color_group_range)

    groupings = np.zeros(shape=(len(rgb_vals)**3, 3), dtype=np.uint8)
    frequencies = np.zeros(shape=len(rgb_vals)**3, dtype=np.int32)
    added = 0

    img = img.astype(dtype=np.int16)
    for r in rgb_vals:
        for g in rgb_vals:
            for b in rgb_vals:

                condition = (
                    (abs(img[:, 0] - r) < color_group_range / 2) &
                    (abs(img[:, 1] - g) < color_group_range / 2) &
                    (abs(img[:, 2] - b) < color_group_range / 2)
                    )
                matching = img[condition]
                if not matching.any():
                    continue

                group_color = np.average(matching, axis=0)
                group_color = cv2.cvtColor(
                    np.array([[group_color]], dtype=np.uint8), cv2.COLOR_RGB2HLS
                ).reshape(3)

                groupings[added, :] = group_color
                frequencies[added] = len(matching)
                added += 1

                img = img[~condition]
                if len(img) <= 1:
                    break
    groupings = groupings[0:added + 1]
    frequencies = frequencies[0:added + 1]

    groupings = np.column_stack((groupings, frequencies))
    groupings = groupings[groupings[:, 3].argsort()][::-1]

    return groupings[:, 0:3], groupings[:, 3]


def get_primary_color(img: np.ndarray, frequencies, avg_color: np.ndarray, total_pixels: int):
    """
    Inputs: img - Image HLS color values (nx3 numpy array)
    frequencies - Image color frequency (n length np array)
    avg_color - Image average HLS color (1x3 np array)
    total_pixels - Amount of pixels in original image after resize (int)
    Outputs: RGB and HLS color values of primary color (1x3 np arrays)
    """

    # Convert to float to do math
    img = img.astype(dtype=np.float32)
    frequencies = frequencies.astype(dtype=np.float32)

    # Initialize NP arrays to group colors into
    frequencies = frequencies[frequencies > 1]
    img = img[0:len(frequencies), :]

    # ADJUSTMENT PARAMETERS
    frequency_threshold = 0.03
    frequency_weight = 1.2
    saturation_min_threshold = 0.05*255
    saturation_max_threshold = 0.65*255
    saturation_weight = 1
    luminosity_min_threshold = 0.2*255
    luminosity_max_threshold = 0.3*255
    luminosity_weight = 0.25
    hue_difference_threshold = 50
    hue_difference_weight = 0.6
    # hue_priority = "[equation to boost magenta color family and reduce tan]"
    # hue_weight = 0

    # Normalized frequency score. Higher frequency results in higher score
    frequency_score = frequency_weight * (
        1 - np.power(
            np.full(shape=frequencies.shape, fill_value=1.0005),
            (total_pixels*frequency_threshold - frequencies)
        )
    )

    # Normalized saturation score. Being between saturation min and max results in a higher score
    saturation_score = (
            1 - 4 / (saturation_max_threshold-saturation_min_threshold)**2 *
            (img[:, 2] - (saturation_min_threshold + saturation_max_threshold)/2)**2
    )
    adjustment = 2 if max(saturation_score) < 0 else 0
    saturation_score = saturation_weight * (adjustment + saturation_score/abs(max(saturation_score)))

    # Normalized luminosity score. Being between luminosity min and max results in a higher score
    luminosity_score = (
            1 - 4 / (luminosity_max_threshold-luminosity_min_threshold)**2 *
            (img[:, 1] - (luminosity_min_threshold + luminosity_max_threshold)/2)**2
    )
    adjustment = 2 if max(luminosity_score) < 0 else 0
    luminosity_score = luminosity_weight * (adjustment + luminosity_score/abs(max(luminosity_score)))

    # Normalized hue score. A lower hue difference than the average image hue results in a higher score
    hue_difference_score = (
            1 - 1 / hue_difference_threshold**2 *
            (img[:, 0] - avg_color[0])**2
    )
    adjustment = 2 if max(hue_difference_score) < 0 else 0
    hue_difference_score = hue_difference_weight * (adjustment + hue_difference_score)/abs(max(hue_difference_score))

    final_score = frequency_score + saturation_score + luminosity_score + hue_difference_score

    idx = final_score.argmax()
    score = final_score[idx]
    primary_color_hls = img[idx, :].astype(np.uint8)

    # print(color)
    # print(primary_color)
    print(f"Frequency Score: {frequency_score[idx]}")
    print(f"Saturation Score: {saturation_score[idx]}")
    print(f"Luminosity Score: {luminosity_score[idx]}")
    print(f"Hue Difference Score: {hue_difference_score[idx]}")
    # print(score)
    # print(frequencies[idx])

    # Correction if best color is outside acceptable luminosity range
    if primary_color_hls[1] < luminosity_min_threshold:
        primary_color_hls[1] *= luminosity_min_threshold / primary_color_hls[1]
    elif primary_color_hls[1] > luminosity_max_threshold:
        primary_color_hls[1] *= luminosity_max_threshold / primary_color_hls[1]

    primary_color_rgb = cv2.cvtColor(np.array([[primary_color_hls]], dtype=np.uint8), cv2.COLOR_HLS2RGB)

    return primary_color_rgb.reshape(3), primary_color_hls


def get_accent_color(img: np.ndarray, frequencies, avg_color: np.ndarray, total_pixels: int, primary_color: np.ndarray):
    """
    Inputs: img - Image HLS color values (nx3 numpy array)
    frequencies - Image color frequency (n length np array)
    avg_color - Image average HLS color (1x3 np array)
    total_pixels - Amount of pixels in original image after resize (int)
    primary_color - Output of get_primary color (1x3 np array)
    Outputs: RGB and HLS color values of accent color (1x3 np arrays)
    """

    # Convert to float to do math
    img = img.astype(dtype=np.float32)
    frequencies = frequencies.astype(dtype=np.float32)

    # Initialize NP arrays to group colors into
    frequencies = frequencies[frequencies > 1]
    img = img[0:len(frequencies), :]

    # ADJUSTMENT PARAMETERS
    frequency_threshold = 0.02
    frequency_weight = 1.2
    saturation_min_threshold = 0.55*255
    saturation_max_threshold = 0.95*255
    saturation_weight = 2
    luminosity_min_threshold = 0.5*255
    luminosity_max_threshold = 0.75*255
    luminosity_weight = 0.5
    hue_difference_threshold = 50
    hue_difference_weight = 1
    # hue_priority = "[equation to boost magenta color family and reduce tan]"
    # hue_weight = 0

    # Normalized frequency score. Higher frequency results in higher score
    frequency_score = frequency_weight * (
        1 - np.power(
            np.full(shape=frequencies.shape, fill_value=1.0005),
            (total_pixels*frequency_threshold - frequencies)
        )
    )

    # Normalized saturation score. Being between saturation min and max results in a higher score
    saturation_score = (
            1 - 4 / (saturation_max_threshold-saturation_min_threshold)**2 *
            (img[:, 2] - (saturation_min_threshold + saturation_max_threshold)/2)**2
    )
    adjustment = 2 if max(saturation_score) < 0 else 0
    saturation_score = saturation_weight * (adjustment + saturation_score/abs(max(saturation_score)))

    # Normalized luminosity score. Being between luminosity min and max results in a higher score
    luminosity_score = (
            1 - 4 / (luminosity_max_threshold-luminosity_min_threshold)**2 *
            (img[:, 1] - (luminosity_min_threshold + luminosity_max_threshold)/2)**2
    )
    adjustment = 2 if max(luminosity_score) < 0 else 0
    luminosity_score = luminosity_weight * (adjustment + luminosity_score/abs(max(luminosity_score)))

    # Normalized hue score. A higher hue difference than the primary image hue results in a higher score, capped at 1

    hue_diff = np.abs(img[:, 0] - primary_color[0])
    hue_diff = np.where(
        hue_diff > 90,
        180 - hue_diff,
        hue_diff
    )
    print(hue_diff)

    hue_difference_score = hue_difference_weight * np.minimum(
            1 / hue_difference_threshold**2 * hue_diff**2 - 1,
            1
    )

    final_score = frequency_score + saturation_score + luminosity_score + hue_difference_score

    idx = final_score.argmax()
    score = final_score[idx]
    accent_color_hls = img[idx, :].astype(np.uint8)

    # print(color)
    # print(accent_color)
    print(f"Frequency Score: {frequency_score[idx]}")
    print(f"Saturation Score: {saturation_score[idx]}")
    print(f"Luminosity Score: {luminosity_score[idx]}")
    print(f"Hue Difference Score: {hue_difference_score[idx]}")
    # print(score)
    # print(frequencies[idx])

    # Correction if best color is outside acceptable luminosity range
    if accent_color_hls[1] < luminosity_min_threshold:
        accent_color_hls[1] *= luminosity_min_threshold / accent_color_hls[1]
    elif accent_color_hls[1] > luminosity_max_threshold:
        accent_color_hls[1] *= luminosity_max_threshold / accent_color_hls[1]

    accent_color_rgb = cv2.cvtColor(np.array([[accent_color_hls]], dtype=np.uint8), cv2.COLOR_HLS2RGB)

    return accent_color_rgb.reshape(3), accent_color_hls


def main():
    filename = "Static/firefox_bkgd.jpg"
    img_rgb, img_hls = read_image(filename=filename)

    total_pixels = len(img_rgb)
    avg_color = np.average(img_rgb, axis=0).astype(np.uint8)
    avg_color_hls = cv2.cvtColor(np.array([[avg_color]]), cv2.COLOR_RGB2HLS).reshape(3)
    print(f"Average Color (RGB): {avg_color}")
    print(f"Average Color (HLS): {avg_color_hls}")
    print(f"Image Pixels: {total_pixels}")

    img_hls, frequencies = get_color_frequency(img=img_rgb, color_group_range=15)

    primary_color, primary_color_hls = get_primary_color(
        img=img_hls, frequencies=frequencies, avg_color=avg_color_hls, total_pixels=total_pixels
    )
    print(f"Primary Color (RGB): {primary_color}")
    print(f"Primary Color (HLS): {primary_color_hls}")

    accent_color, accent_color_hls = get_accent_color(
        img=img_hls, frequencies=frequencies, avg_color=avg_color_hls,
        total_pixels=total_pixels, primary_color=primary_color_hls
    )
    print(f"Accent Color (RGB): {accent_color}")
    print(f"Accent Color (HLS): {accent_color_hls}")

    full_image = cv2.imread(filename=filename, flags=cv2.IMREAD_UNCHANGED)

    avg_color_bgr = cv2.cvtColor(np.array([[avg_color]], dtype=np.uint8), cv2.COLOR_RGB2BGR).reshape(1, 3)
    primary_color_bgr = cv2.cvtColor(np.array([[primary_color]], dtype=np.uint8), cv2.COLOR_RGB2BGR).reshape(1, 3)
    accent_color_bgr = cv2.cvtColor(np.array([[accent_color]], dtype=np.uint8), cv2.COLOR_RGB2BGR).reshape(1, 3)
    cv2.imshow("Original", full_image)
    cv2.imshow("Average", np.full(shape=(100, 100, 3), fill_value=avg_color_bgr, dtype=np.uint8))
    cv2.imshow("Primary", np.full(shape=(100, 100, 3), fill_value=primary_color_bgr, dtype=np.uint8))
    cv2.imshow("Accent", np.full(shape=(100, 100, 3), fill_value=accent_color_bgr, dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
