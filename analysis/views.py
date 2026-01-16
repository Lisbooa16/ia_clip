from django.shortcuts import render

from .services.viral_analysis import build_analysis


def analysis_page(request):
    result = None
    url = ""
    error = ""

    if request.method == "POST":
        url = request.POST.get("url", "").strip()

        if url:
            result = build_analysis(url).to_dict()
        else:
            error = "Informe uma URL válida."

    return render(
        request,
        "analysis/analysis.html",
        {
            "url": url,
            "result": result,
            "error": error,
        },
    )
