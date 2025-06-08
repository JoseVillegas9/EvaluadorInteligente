using Microsoft.ML.Data;

namespace EvaluadorInteligente.Models.Recommendation
{
    public class ProductRating
    {
        // Columna 0: Id del usuario
        [LoadColumn(0)]
        public string UserId;

        // Columna 1: Id del producto
        [LoadColumn(1)]
        public string ProductId;

        // Columna 2: Calificación (float)
        [LoadColumn(2)]
        public float Label;
    }
}