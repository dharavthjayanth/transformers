export const queryLLM = async (userQuery) => {
  const token = localStorage.getItem("token");

  const response = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ query: userQuery }),
  });

  if (!response.ok) {
    throw new Error("Failed to get response from LLM");
  }

  const data = await response.json();
  return data.response;
};
