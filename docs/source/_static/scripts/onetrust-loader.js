(function () {
  // Use the OneTrust test script because nvidia-merlin.github.io is not a
  // production NVIDIA web property registered for the production script.
  const dataDomainScript = "3e2b62ff-7ae7-4ac5-87c8-d5949ecafff5-test";
  const stubSrc = "https://cdn.cookielaw.org/scripttemplates/otSDKStub.js";
  const otCustomSrc =
    "https://images.nvidia.com/aem-dam/Solutions/ot-js/ot-custom.js";

  window.OptanonWrapper = function () {
    window.dispatchEvent(new Event("bannerLoaded"));
  };

  const stub = document.createElement("script");
  stub.src = stubSrc;
  stub.type = "text/javascript";
  stub.charset = "UTF-8";
  stub.setAttribute("data-document-language", "true");
  stub.setAttribute("data-domain-script", dataDomainScript);
  stub.onload = function () {
    const custom = document.createElement("script");
    custom.src = otCustomSrc;
    custom.type = "text/javascript";
    document.head.appendChild(custom);
  };
  document.head.appendChild(stub);
})();
