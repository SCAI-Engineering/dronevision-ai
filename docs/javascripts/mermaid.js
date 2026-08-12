document$.subscribe(async () => {
  const dark = document.body.getAttribute("data-md-color-scheme") === "dronevision-dark";
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    themeVariables: {
      background: dark ? "#071827" : "#f4f8fa",
      primaryColor: dark ? "#103149" : "#dceff4",
      primaryTextColor: dark ? "#eaf5f8" : "#102838",
      primaryBorderColor: "#5bd7f5",
      lineColor: dark ? "#6f8fa2" : "#3f6f82",
      secondaryColor: dark ? "#163a4f" : "#e7f1f4",
      secondaryTextColor: dark ? "#eaf5f8" : "#102838",
      tertiaryColor: "#ffb454",
      tertiaryTextColor: "#071827",
      clusterBkg: dark ? "#0b2234" : "#eaf3f6",
      clusterBorder: "#397b92",
      fontFamily: "IBM Plex Sans, sans-serif"
    }
  });
  const diagrams = [...document.querySelectorAll("pre.mermaid")].map((block) => {
    const diagram = document.createElement("div");
    diagram.className = "mermaid";
    diagram.textContent = block.textContent;
    block.replaceWith(diagram);
    return diagram;
  });
  if (diagrams.length) {
    await mermaid.run({ nodes: diagrams });
  }
});
