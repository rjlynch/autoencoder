class AutoEncoder < Torch::NN::Module
  INPUT_DIM = 784

  attr_reader :encoder, :decoder

  def initialize
    super

    @encoder = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(INPUT_DIM, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, 256),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(256, 128),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(128, 64),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(64, 32),
    )

    @decoder = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(32, 64),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(64, 128),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(128, 256),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(256, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, INPUT_DIM),
    )
  end

  def forward(image)
    latent = forward_encoder(image)
    forward_decoder(latent)
  end

  def forward_encoder(image)
    # Flatten (28px X 28px => 784)
    image_vector = image.view([-1, INPUT_DIM])

    @encoder.forward(image_vector)
  end

  def forward_decoder(latent)
    reconstructed = @decoder.forward(latent)

    # reshape back to image (784 => 28px X 28px)
    reconstructed.view([-1, 1, 28, 28])
  end
end
