document$.subscribe(async () => {
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    themeVariables: {
      background: "#071827",
      primaryColor: "#103149",
      primaryTextColor: "#eaf5f8",
      primaryBorderColor: "#5bd7f5",
      lineColor: "#6f8fa2",
      secondaryColor: "#163a4f",
      secondaryTextColor: "#eaf5f8",
      tertiaryColor: "#ffb454",
      tertiaryTextColor: "#071827",
      clusterBkg: "#0b2234",
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
