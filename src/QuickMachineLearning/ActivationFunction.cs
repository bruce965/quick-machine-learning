using System;

namespace QuickMachineLearning
{
    public enum ActivationFunction
    {
        None,
        BinaryThreshold,
        LinearThreshold,
        Sigmoid,
    }

    public static class ActivationFunctionExtensions
    {
        public static Func<float, float> AsFunction(this ActivationFunction activationFunction) => activationFunction switch
        {
            ActivationFunction.None => v => v,
            ActivationFunction.BinaryThreshold => v => v >= 0f ? 1f : 0f,
            ActivationFunction.LinearThreshold => v => v <= 0f ? 0f : v >= 1f ? 1f : v,
            ActivationFunction.Sigmoid => v => 1f / (1f + MathF.Exp(-v)),
            _ => throw new ArgumentException($"Invalid activation function '{activationFunction}'.", nameof(activationFunction))
        };
    }
}
