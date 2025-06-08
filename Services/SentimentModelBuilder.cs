using System;
using System.IO;
using Microsoft.ML;
using EvaluadorInteligente.Models.Sentiment;

namespace EvaluadorInteligente.Services
{
    public static class SentimentModelBuilder
{
    public static void TrainAndSave(string contentRootPath)
    {
        var mlContext = new MLContext();

        // Ahora usamos contentRootPath en vez de Environment.CurrentDirectory
        var dataPath = Path.Combine(contentRootPath, "MLModels", "Sentiment", "sentiment-data.tsv");
        var data = mlContext.Data
            .LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, separatorChar: '\t');

        var pipeline = mlContext.Transforms.Text
            .FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification
                .Trainers.SdcaLogisticRegression(nameof(SentimentData.Label), "Features"));

        var model = pipeline.Fit(data);
        var modelPath = Path.Combine(contentRootPath, "MLModels", "Sentiment", "SentimentModel.zip");
        mlContext.Model.Save(model, data.Schema, modelPath);

        Console.WriteLine($"SentimentModel guardado en: {modelPath}");
    }
}

}
