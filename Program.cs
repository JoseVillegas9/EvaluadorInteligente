using EvaluadorInteligente.Services;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllersWithViews();

var app = builder.Build();

// Obtiene la carpeta raíz del proyecto donde están MLModels/
var contentRoot = app.Environment.ContentRootPath;

// Entrena y guarda los modelos usando contentRoot
SentimentModelBuilder.TrainAndSave(contentRoot);
RecommendationModelBuilder.TrainAndSave(contentRoot);

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

// Ahora arranca en Sentiment/Analyze por defecto
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Sentiment}/{action=Analyze}/{id?}");

app.Run();
