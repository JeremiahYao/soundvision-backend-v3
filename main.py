frame_id = 0
    process_every_n_frames = 3  # INCREASE THIS TO REDUCE LAG (e.g., 5 or 10)
    
    last_top_risk = None
    last_message = "Path clear."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_id % process_every_n_frames == 0:
            # Only run AI every N frames
            detections = detector.detect(frame)
            analyzed = spatial.analyze(detections, width, height, frame_id)
            last_top_risk = engine.evaluate(analyzed)
            last_message = guidance.generate(last_top_risk) if last_top_risk else "Path clear."

        # Still draw and write every frame so the video looks smooth
        if last_top_risk:
            # Draw overlay using the 'remembered' risk from the last AI pass
            pass
