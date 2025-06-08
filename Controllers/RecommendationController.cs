using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using EvaluadorInteligente.Models.Recommendation;
using System.Collections.Generic;
using System.Linq;

namespace EvaluadorInteligente.Controllers
{
    public class RecommendationController : Controller
    {
        private readonly MLContext _mlContext = new MLContext();
        private ITransformer _model;
        private PredictionEngine<ProductRating, ProductPrediction> _predictor;
        private List<string> _allProducts;

        public RecommendationController()
        {
            _model = _mlContext.Model.Load("MLModels/Recommendation/RecModel.zip", out _);
            _predictor = _mlContext.Model
                .CreatePredictionEngine<ProductRating, ProductPrediction>(_model);
            _allProducts = System.IO.File.ReadAllLines("MLModels/Recommendation/ratings-data.csv")
                                 .Skip(1)
                                 .Select(r => r.Split(',')[1])
                                 .Distinct()
                                 .ToList();
        }

        [HttpGet]
        public IActionResult Recommend() => View();

        [HttpPost]
        public IActionResult Recommend(string userId)
        {
            var recommendations = new List<(string ProductId, float Score)>();
            foreach (var prod in _allProducts)
            {
                var pred = _predictor.Predict(
                    new ProductRating { UserId = userId, ProductId = prod });
                recommendations.Add((prod, pred.Score));
            }
            var top = recommendations.OrderByDescending(r => r.Score).Take(5);
            return View("RecommendResult", top);
        }
    }
}
