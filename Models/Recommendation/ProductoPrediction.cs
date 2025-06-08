using Microsoft.ML.Data;

namespace EvaluadorInteligente.Models.Recommendation
{
    public class ProductPrediction
    {
        // El Score que devuelve el modelo de MatrixFactorization
        public float Score { get; set; }
    }
}