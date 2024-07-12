
def initialize_models(num_series, features, hidden_size, num_layers, num_directions, prediction_length, spatial_feature_size, config):
    from .encoder import MultiSeriesEncoder
    encoder = MultiSeriesEncoder(features, hidden_size, num_layers, num_directions, prediction_length)

    if config.get('spatial', {}).get('spatial_use', True):
        from .decoder import MultiSeriesDecoder
        decoder = MultiSeriesDecoder(hidden_size, num_series, num_layers, num_directions, prediction_length, spatial_feature_size)
    else:
        from .decoder_no_spatial import MultiSeriesDecoderNoSpatial
        decoder = MultiSeriesDecoderNoSpatial(hidden_size, num_series, num_layers, num_directions, prediction_length, spatial_feature_size)
    return encoder, decoder