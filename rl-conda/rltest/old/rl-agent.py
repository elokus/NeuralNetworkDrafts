


#learns from observation wich action to take. Input is an 48 dimension matrix
def get_model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    n_actions = output_dims

    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(, activation='softmax', name='predictions')(x)

    # Define model
    model = Model(inputs=[state_input],outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model


#Critic Model learn to evaluate if the action taken by the Actor led our env to be in a better state based on the rewards from the last timesteps
def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh')(x)

    # Define model
    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model
