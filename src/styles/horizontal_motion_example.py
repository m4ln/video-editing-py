def mosh_frames(frames):
    for frame in frames:
        if not frame:
            continue

        for row in frame:
            for col in row:
                # col contains the horizontal and vertical components of the vector
                col[0] = 0  # Set horizontal component to 0 for vertical motion only
                # col[1] = 0  # Set vertical component to 0 for horizontal motion only

    return frames
