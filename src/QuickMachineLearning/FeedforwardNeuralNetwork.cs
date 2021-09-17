using System;
using System.IO;
using System.Text;

namespace QuickMachineLearning
{
    public class FeedforwardNeuralNetwork : ICloneable
    {
        public struct Layer
        {
            public FeedforwardNeuralNetwork NeuralNetwork { get; }
            public int LayerIndex { get; }
            public bool IsInputLayer => LayerIndex == 0;
            public bool IsOutputLayer => LayerIndex == NeuralNetwork.LayersCount - 1;

            public int NeuronsCount => NeuralNetwork._layerNeurons[LayerIndex];

            public Neuron this[int neuronIndex] => new Neuron(this, neuronIndex);

            public ActivationFunction ActivationFunction
            {
                get
                {
                    if (LayerIndex == 0)
                        return default;

                    return NeuralNetwork._layerActivationFunctions[LayerIndex - 1];
                }
                set
                {
                    if (LayerIndex == 0)
                        throw new NotSupportedException("Input layer does not support an activation function.");

                    NeuralNetwork._layerActivationFunctions[LayerIndex - 1] = value;
                }
            }

            internal Layer(FeedforwardNeuralNetwork neuralNetwork, int layerIndex)
            {
                if (layerIndex < 0 || layerIndex > neuralNetwork._layerNeurons.Length - 1)
                    throw new IndexOutOfRangeException("Invalid layer index.");

                NeuralNetwork = neuralNetwork;
                LayerIndex = layerIndex;
            }
        }

        public struct Neuron
        {
            public Layer Layer { get; }
            public int NeuronIndex { get; }
            
            public bool IsInputNeuron => Layer.IsInputLayer;
            public bool IsOutputNeuron => Layer.IsOutputLayer;

            public Span<float> Weights => Layer.LayerIndex == 0 ? Span<float>.Empty : Layer.NeuralNetwork._weightsAndBiases.AsSpan(
                Layer.NeuralNetwork._previousLayerWeights[Layer.LayerIndex] + NeuronIndex * Layer.NeuralNetwork._layerNeurons[Layer.LayerIndex - 1],
                Layer.NeuralNetwork._layerNeurons[Layer.LayerIndex - 1]
            );

            public float Bias
            {
                get
                {
                    if (Layer.LayerIndex == 0)
                        return 0;

                    return Layer.NeuralNetwork._weightsAndBiases[
                        Layer.NeuralNetwork._weightsCount + Layer.NeuralNetwork._previousLayerBiases[Layer.LayerIndex] + NeuronIndex
                    ];
                }
                set
                {
                    if (Layer.LayerIndex == 0)
                        throw new NotSupportedException("Input neurons do not support bias.");

                    Layer.NeuralNetwork._weightsAndBiases[
                        Layer.NeuralNetwork._weightsCount + Layer.NeuralNetwork._previousLayerBiases[Layer.LayerIndex] + NeuronIndex
                    ] = value;
                }
            }

            internal Neuron(Layer layer, int neuronIndex)
            {
                if (neuronIndex < 0 || neuronIndex > layer.NeuronsCount - 1)
                    throw new IndexOutOfRangeException("Invalid neuron index.");

                Layer = layer;
                NeuronIndex = neuronIndex;
            }
        }

        const int MagicNumber = 0x46464e31;  // "QFN1"

        public int LayersCount => _layerNeurons.Length;

        public Layer this[int layerIndex] => new Layer(this, layerIndex);

        public Layer InputLayer => this[0];
        public Layer OutputLayer => this[_layerNeurons.Length - 1];

        /// <summary>
        /// Number of neurons for each layer (input layer, hidden layers, output layer).
        /// </summary>
        readonly int[] _layerNeurons;
        readonly int[] _previousLayerWeights;  // used to index weights
        readonly int[] _previousLayerBiases;  // used to index biases

        /// <summary>
        /// Synaptic weights (hidden layers, output layer), followed by neuron biases (hidden layers, output layer).
        /// </summary>
        readonly float[] _weightsAndBiases;

        readonly ActivationFunction[] _layerActivationFunctions;

        readonly int _weightsCount;

        readonly int _biggestHiddenLayerSize;

        public FeedforwardNeuralNetwork(ActivationFunction activationFunction, params int[] layerNeurons)
            : this(activationFunction, layerNeurons.AsSpan()) { }

        public FeedforwardNeuralNetwork(ActivationFunction activationFunction, ReadOnlySpan<int> layerNeurons)
        {
            if (layerNeurons.Length < 2)
                throw new ArgumentException("At least one input and one output layers are required.", nameof(layerNeurons));

            _layerNeurons = layerNeurons.ToArray();

            _layerActivationFunctions = new ActivationFunction[_layerNeurons.Length - 1];
            for (var i = 0 ; i < _layerActivationFunctions.Length; i++)
                _layerActivationFunctions[i] = activationFunction;

            _previousLayerWeights = new int[_layerNeurons.Length];
            _previousLayerBiases = new int[_layerNeurons.Length];

            var biasesCount = 0;
            _weightsCount = 0;

            for (var i = 1; i < _layerNeurons.Length; i++)
            {
                _previousLayerWeights[i] = _weightsCount;
                _weightsCount += _layerNeurons[i - 1] * _layerNeurons[i];

                _previousLayerBiases[i] = biasesCount;
                biasesCount += _layerNeurons[i];

                if (i < _layerNeurons.Length - 1 && _layerNeurons[i] > _biggestHiddenLayerSize)
                    _biggestHiddenLayerSize = _layerNeurons[i];
            }

            _weightsAndBiases = new float[_weightsCount + biasesCount];
        }

        public void Serialize(Stream stream)
        {
            var writer = new BinaryWriter(stream, Encoding.ASCII, true);

            writer.Write((int)MagicNumber);

            var hiddenLayersCount = _layerNeurons.Length - 2;
            writer.Write(checked((ushort)hiddenLayersCount));

            for (var i = 0; i < _layerActivationFunctions.Length; i++)
                writer.Write((byte)_layerActivationFunctions[i]);

            for (var i = 0; i < _layerNeurons.Length; i++)
                writer.Write((int)_layerNeurons[i]);

            for (var i = 0; i < _weightsAndBiases.Length; i++)
                writer.Write((float)_weightsAndBiases[i]);
        }

        public static FeedforwardNeuralNetwork Deserialize(Stream stream)
        {
            var reader = new BinaryReader(stream, Encoding.ASCII, true);

            var magicNumber = reader.ReadInt32();
            if (magicNumber != MagicNumber)
                throw new FormatException("Invalid data (magic number mismatch).");

            var hiddenLayersCount = reader.ReadUInt16();
            var layersCount = hiddenLayersCount + 2;

            Span<ActivationFunction> layerActivationFunctions = stackalloc ActivationFunction[layersCount - 1];
            for (var i = 0; i < layerActivationFunctions.Length; i++)
                layerActivationFunctions[i] = (ActivationFunction)reader.ReadByte();

            Span<int> layerNeurons = stackalloc int[layersCount];
            for (var i = 0; i < layerNeurons.Length; i++)
                layerNeurons[i] = reader.ReadInt32();

            var neuralNetwork = new FeedforwardNeuralNetwork(layerActivationFunctions[0], layerNeurons);
            for (var layerIndex = 2; layerIndex < layersCount; layerIndex++)
            {
                var layer = neuralNetwork[layerIndex];
                layer.ActivationFunction = layerActivationFunctions[layerIndex - 1];
            }
            
            for (var i = 0; i < neuralNetwork._weightsAndBiases.Length; i++)
                neuralNetwork._weightsAndBiases[i] = reader.ReadSingle();

            return neuralNetwork;
        }

        public FeedforwardNeuralNetwork Clone()
        {
            var stream = new MemoryStream(
                + sizeof(int)
                + sizeof(ushort)
                + _layerActivationFunctions.Length * sizeof(byte)
                + _layerNeurons.Length * sizeof(int)
                + _weightsAndBiases.Length * sizeof(float)
            );

            Serialize(stream);

            stream.Position = 0;

            return Deserialize(stream);
        }

        object ICloneable.Clone()
            => Clone();

        public void Compute(ReadOnlySpan<float> inputs, Span<float> outputs)
        {
            if (inputs.Length != _layerNeurons[0])
                throw new ArgumentException("Number of inputs does not match.", nameof(inputs));

            if (outputs.Length != _layerNeurons[^1])
                throw new ArgumentException("Number of outputs does not match.", nameof(outputs));

            Span<float> currentLayerOutputs = stackalloc float[_biggestHiddenLayerSize];
            Span<float> buffer = stackalloc float[_biggestHiddenLayerSize];

            // HACK: see [HACK_REF], cannot use local 'currentLayerOutputs' in this context because it may expose referenced variables outside of their declaration scope.
            ReadOnlySpan<float> previousLayerOutputs = currentLayerOutputs;
            previousLayerOutputs = inputs;

            var weightIndex = 0;
            var biasIndex = _weightsCount;

            for (var layerIndex = 1; layerIndex < _layerNeurons.Length; layerIndex++)
            {
                if (layerIndex == _layerNeurons.Length - 1)
                    currentLayerOutputs = outputs;

                var activate = _layerActivationFunctions[layerIndex - 1].AsFunction();

                for (var i = 0; i < _layerNeurons[layerIndex]; i++)
                {
                    var bias = _weightsAndBiases[biasIndex++];

                    var sum = bias;

                    for (var j = 0; j < _layerNeurons[layerIndex - 1]; j++)
                    {
                        var input = previousLayerOutputs[j];
                        var weight = _weightsAndBiases[weightIndex++];

                        sum += input * weight;
                    }

                    currentLayerOutputs[i] = activate(sum);
                }

                previousLayerOutputs = currentLayerOutputs;  // [HACK_REF]

                var exchange = currentLayerOutputs;
                currentLayerOutputs = buffer;
                buffer = exchange;
            }
        }
    }
}
