def onion_skin_radar(current_image_path, image_paths, n_previous=100, alpha_decay=0.3, show=True):
    current_index = image_paths.index(Path(current_image_path))
    start_index = max(0, current_index - n_previous)
    
    # Base image (the current one)
    base_image = Image.open(image_paths[current_index]).convert("RGBA")
    base_image = base_image.resize((300, 300), Image.NEAREST)
    
    composite = base_image.copy()
    
    # Previous images (from start_index to current_index, excluding current)
    previous_images = image_paths[start_index:current_index]
    
    for i, prev_path in enumerate(previous_images):
        frame_distance = len(previous_images) - i
        alpha = alpha_decay ** frame_distance
        
        prev_image = Image.open(prev_path).convert("RGBA")
        prev_image = prev_image.resize((300, 300), Image.NEAREST)
        
        alpha_channel = prev_image.getchannel('A')
        alpha_channel = alpha_channel.point(lambda x: int(x * alpha))
        prev_image.putalpha(alpha_channel)
        
        composite = Image.alpha_composite(composite, prev_image)
    
    return composite.convert('RGB')
