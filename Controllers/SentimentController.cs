using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using EvaluadorInteligente.Models.Sentiment;

namespace EvaluadorInteligente.Controllers
{
    public class SentimentController : Controller
    {
        private static PredictionEngine<SentimentData, SentimentPrediction> _predictor;

        public SentimentController()
        {
            if (_predictor == null)
            {
                var mlContext = new MLContext();
                var model = mlContext.Model.Load("MLModels/Sentiment/SentimentModel.zip", out _);
                _predictor = mlContext.Model
                    .CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            }
        }

        [HttpGet]
        public IActionResult Analyze() => View();

        [HttpPost]
        public IActionResult Analyze(string text)
        {
            var input = new SentimentData { Text = text };
            var result = _predictor.Predict(input);
            ViewBag.Prediction  = result.Prediction ? "Positivo" : "Negativo";
            ViewBag.Probability = result.Probability;
            return View("AnalyzeResult");
        }
    }
}
