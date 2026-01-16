# analysis/views.py
from django.shortcuts import render
from .services.youtube import search_youtube_videos
from .utils import calculate_engagement, calculate_viral_score


def analysis_page(request):
    query = ""
    results = []

    if request.method == "POST":
        query = request.POST.get("query")

        if query:
            videos = search_youtube_videos(query)
            print(videos)

            for v in videos:
                if not v:
                    continue
                engagement = calculate_engagement(
                    v.get("views", 0),
                    v.get("likes"),
                    v.get("comments"),
                )

                viral_score = calculate_viral_score(
                    v.get("views", 0),
                    engagement,
                    v.get("duration"),
                )

                results.append({
                    "title": v["title"],
                    "url": v["url"],
                    "views": v["views"],
                    "duration": v["duration"],
                    "engagement": round(engagement, 4),
                    "viral_score": viral_score,
                })

            results.sort(key=lambda x: x["viral_score"], reverse=True)
            results = results[:10]

    return render(request, "analysis/analysis.html", {
        "query": query,
        "results": results
    })
