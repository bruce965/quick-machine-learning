using System;

namespace QuickMachineLearning
{
    public static class FeedforwardNeuralNetworkExtensions
    {
        static readonly Random random = new Random();

        public static void RandomizeWeightsAndBiases(this FeedforwardNeuralNetwork neuralNetwork)
        {
            for (var layerIndex = 0; layerIndex < neuralNetwork.LayersCount; layerIndex++)
            {
                var layer = neuralNetwork[layerIndex];

                for (var neuronIndex = 0; neuronIndex < layer.NeuronsCount; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];

                    var weights = neuron.Weights;
                    for (var i = 0; i < weights.Length; i++)
                        weights[i] = (float)random.NextDouble();

                    if (layerIndex != 0)
                        neuron.Bias = (float)random.NextDouble();
                }
            }
        }

        public static void MutateRandomly(this FeedforwardNeuralNetwork neuralNetwork, float weight, int count)
        {
            var totalNeuronsCount = 0;
            for (var layerIndex = 1; layerIndex < neuralNetwork.LayersCount; layerIndex++)
                totalNeuronsCount += neuralNetwork[layerIndex].NeuronsCount;

            for (var mutation = 0; mutation < count; mutation++)
            {
                var mutateNeuronIndex = Convert.ToInt32(Math.Floor(random.NextDouble() * totalNeuronsCount));

                int layerFirstNeuronIndex;
                var neuronsCountSoFar = 0;
                for (var layerIndex = 1; layerIndex < neuralNetwork.LayersCount; layerIndex++)
                {
                    layerFirstNeuronIndex = neuronsCountSoFar;
                    neuronsCountSoFar += neuralNetwork[layerIndex].NeuronsCount;

                    if (neuronsCountSoFar > mutateNeuronIndex)
                    {
                        var neuron = neuralNetwork[layerIndex][mutateNeuronIndex - layerFirstNeuronIndex];
                        var mutateWeightIndex = Convert.ToInt32(Math.Floor((float)random.NextDouble() * (neuron.Weights.Length + 1)));
                        var mutationStrength = ((float)random.NextDouble() * 2f - 1f) * weight;

                        if (mutateWeightIndex == 0)
                            neuron.Bias += mutationStrength;
                        else
                            neuron.Weights[mutateWeightIndex - 1] += mutationStrength;

                        break;
                    }
                }
            }
        }
    
        public static void ClampWeights(this FeedforwardNeuralNetwork neuralNetwork, float min, float max)
        {
            for (var layerIndex = 0; layerIndex < neuralNetwork.LayersCount; layerIndex++)
            {
                var layer = neuralNetwork[layerIndex];

                for (var neuronIndex = 0; neuronIndex < layer.NeuronsCount; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];

                    var weights = neuron.Weights;
                    for (var i = 0; i < weights.Length; i++)
                        weights[i] = Math.Clamp(weights[i], min, max);
                }
            }
        }

        public static void ClampBiases(this FeedforwardNeuralNetwork neuralNetwork, float min, float max)
        {
            for (var layerIndex = 1; layerIndex < neuralNetwork.LayersCount; layerIndex++)
            {
                var layer = neuralNetwork[layerIndex];

                for (var neuronIndex = 0; neuronIndex < layer.NeuronsCount; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];
                    neuron.Bias = Math.Clamp(neuron.Bias, min, max);
                }
            }
        }
    }
}
