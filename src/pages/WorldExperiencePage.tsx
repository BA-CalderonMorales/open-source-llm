import { useParams, Navigate } from "react-router-dom";
import { ExperienceProvider } from "@/context/ExperienceContext";
import ExperienceContent from "@/components/experience/ExperienceContent";
import { useWorldBySlug } from "@/hooks/useWorlds";
import LoadingOverlay from "@/components/experience/LoadingOverlay";

const WorldExperiencePage = () => {

  const { worldSlug } = useParams<{ worldSlug: string }>();

  const slug = worldSlug
    ? decodeURIComponent(worldSlug).trim().toLowerCase()
    : '';

  const { data: world, isLoading, isError } = useWorldBySlug(slug);

  if (isLoading) {
    return <LoadingOverlay message="Loading world..." theme="night" />;
  }

  if (isError || !world) {
    return <Navigate to="/" replace />;
  }

  return (

    <ExperienceProvider>
      <ExperienceContent initialWorldSlug={slug} />
    </ExperienceProvider>
  
  );

};

export default WorldExperiencePage;
