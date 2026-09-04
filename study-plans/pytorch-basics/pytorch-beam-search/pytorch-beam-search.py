def beam_search(log_probs_fn, start_token, end_token, beam_width, max_len):
    """
    Returns: list of token IDs
    """

    active_beams = [([start_token], 0.0)]
    completed_beams = []

    print(max_len, start_token, end_token)
    step = 0
    while len(active_beams) and step < max_len:
        step += 1
        possible_next_beams = []
        for a_beam, score in active_beams:
            for token, log_prob in enumerate(log_probs_fn(a_beam)):
                possible_next_beams.append((a_beam + [token], score + log_prob))

        top_k_beams = sorted(possible_next_beams, reverse=True, key= lambda t: t[1])[:beam_width]
        new_active_beams = []
        for beam in top_k_beams:
            if beam[0][-1] == end_token:
                completed_beams.append(beam)
            else:
                new_active_beams.append(beam)

        active_beams = new_active_beams

    final_beam = sorted(completed_beams + active_beams, reverse=True, key= lambda t: t[1])[0][0]
    return final_beam if final_beam[-1] != end_token else final_beam[:-1]
