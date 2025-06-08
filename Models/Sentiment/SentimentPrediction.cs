using Microsoft.ML.Data;

namespace EvaluadorInteligente.Models.Sentiment
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
