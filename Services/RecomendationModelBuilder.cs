using System;
using System.IO;
using System.Linq;
using Microsoft.ML;                   // MLContext, ITransformer…
using Microsoft.ML.Data;              // LoadColumn, ColumnName…
using Microsoft.ML.Trainers;          // MatrixFactorizationTrainer
using Microsoft.ML.Recommender;       // mlContext.Recommendation()
using EvaluadorInteligente.Models.Recommendation;

namespace EvaluadorInteligente.Services
{
    public static class RecommendationModelBuilder
    {
        public static void TrainAndSave(string contentRootPath)
        {
            var mlContext = new MLContext();

            // --- DIAGNÓSTICO: listar carpeta Recommendation y comprobación de CSV ---
            var recFolder = Path.Combine(contentRootPath, "MLModels", "Recommendation");
            Console.WriteLine($"[DEBUG] ContentRootPath: {contentRootPath}");
            Console.WriteLine($"[DEBUG] Recommendation folder: {recFolder}");
            if (Directory.Exists(recFolder))
            {
                Console.WriteLine("[DEBUG] Archivos en carpeta Recommendation:");
                foreach (var f in Directory.GetFiles(recFolder))
                    Console.WriteLine("    " + Path.GetFileName(f));
            }
            else
            {
                Console.WriteLine("[DEBUG] ¡La carpeta Recommendation NO existe!");
            }
            var dataPath = Path.Combine(recFolder, "ratings-data.csv");
            Console.WriteLine($"[DEBUG] Ruta CSV: {dataPath}");
            Console.WriteLine($"[DEBUG] File.Exists(csv) = {File.Exists(dataPath)}");
            // ----------------------------------------------------------------------

            // Cargar datos de entrenamiento
            var data = mlContext.Data.LoadFromTextFile<ProductRating>(
                dataPath, hasHeader: true, separatorChar: ',');

            // Pipeline de recomendación
            var pipeline = mlContext.Transforms.Conversion
                    .MapValueToKey("userIdEncoded", nameof(ProductRating.UserId))
                .Append(mlContext.Transforms.Conversion
                    .MapValueToKey("productIdEncoded", nameof(ProductRating.ProductId)))
                .Append(mlContext.Recommendation()
                    .Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options
                    {
                        MatrixColumnIndexColumnName = "userIdEncoded",
                        MatrixRowIndexColumnName    = "productIdEncoded",
                        LabelColumnName             = nameof(ProductRating.Label),
                        NumberOfIterations          = 20,
                        ApproximationRank           = 100
                    }));

            // Entrenamiento y guardado
            var model = pipeline.Fit(data);
            var modelPath = Path.Combine(recFolder, "RecModel.zip");
            mlContext.Model.Save(model, data.Schema, modelPath);
            Console.WriteLine($"RecModel guardado en: {modelPath}");
        }
    }
}
