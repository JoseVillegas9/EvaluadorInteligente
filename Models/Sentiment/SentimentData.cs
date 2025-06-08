using Microsoft.ML.Data;

namespace EvaluadorInteligente.Models.Sentiment
{
    public class SentimentData
    {
        [LoadColumn(0)] public bool Label { get; set; }
        [LoadColumn(1)] public string Text { get; set; }
    }
}
