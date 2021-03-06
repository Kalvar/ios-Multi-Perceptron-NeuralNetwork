Pod::Spec.new do |s|
  s.name         = "KRMLP"
  s.version      = "2.2.3"
  s.summary      = "Deep Learning for multi-layer perceptrons neural network (MLP)."
  s.description  = <<-DESC
                   Machine Learning (マシンラーニング) in this project, it implemented multi-layer perceptrons neural network (ニューラルネットワーク) and Back Propagation Neural Network (BPN). It designed unlimited hidden layers to do the training tasks. This network can be used in products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork.git", :tag => s.version.to_s }
  s.platform     = :ios, '9.0'
  s.requires_arc = true
  s.public_header_files = 'ML/**/*.h'
  s.source_files = 'ML/**/*.{h,m}'
  s.frameworks   = 'Foundation'
end 